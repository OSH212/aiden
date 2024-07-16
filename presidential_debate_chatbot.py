import streamlit as st
import os
import cohere
from litellm import completion
from litellm.utils import trim_messages
import asyncio
import aiohttp
import json
import numpy as np
import sqlite3
from nltk.tokenize import sent_tokenize
import nltk
from loguru import logger
import time
from tqdm import tqdm

nltk.download('punkt', quiet=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'biden_ai' not in st.session_state:
    st.session_state.biden_ai = None
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 15000

with open('voices.txt', 'r') as f:
    VOICES = [line.strip() for line in f]

LLM_MODELS = [
    "openrouter/cognitivecomputations/dolphin-mixtral-8x22b",
    "openrouter/anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet-20240620",
    "anthropic/claude-3-haiku-20240307",
    "custom"
]

class SQLiteBidenAISystem:
    def __init__(self, db_path='bidaji_ai.db'):
        self.co = cohere.Client(api_key=st.session_state.cohere_api_key)
        self.db_path = db_path
        self.embed_model = "embed-english-v3.0"
        self.embed_dim = 1024
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            ''')
            conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def add_documents(self, documents: list[str]):
        new_docs = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for doc in documents:
                cursor.execute("SELECT id FROM documents WHERE content = ?", (doc,))
                if cursor.fetchone() is None:
                    new_docs.append(doc)

        logger.info(f"Found {len(new_docs)} new documents to add.")

        if new_docs:
            batch_size = 90
            total_batches = (len(new_docs) + batch_size - 1) // batch_size
            
            for i in range(0, len(new_docs), batch_size):
                batch = new_docs[i:i+batch_size]
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        logger.info(f"Embedding batch {i//batch_size + 1}/{total_batches} ({len(batch)} documents)")
                        embeddings = self.co.embed(
                            texts=batch,
                            model=self.embed_model,
                            input_type="search_document"
                        ).embeddings
                        logger.info(f"Embedding shape: {np.array(embeddings).shape}")

                        with sqlite3.connect(self.db_path) as conn:
                            cursor = conn.cursor()
                            for doc, emb in zip(batch, embeddings):
                                if len(emb) != self.embed_dim:
                                    logger.error(f"Embedding dimension mismatch: got {len(emb)}, expected {self.embed_dim}")
                                    continue
                                cursor.execute(
                                    "INSERT INTO documents (content, embedding) VALUES (?, ?)",
                                    (doc, np.array(emb, dtype=np.float32).tobytes())
                                )
                            conn.commit()
                        logger.info(f"Added {len(batch)} new documents to the database.")
                        break
                    except cohere.CohereAPIError as e:
                        logger.error(f"Cohere API error: {str(e)}")
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"Retrying in 5 seconds... (Attempt {retry_count + 1}/{max_retries})")
                            time.sleep(5)
                        else:
                            logger.error("Max retries reached. Skipping this batch.")
                    except Exception as e:
                        logger.error(f"Unexpected error: {str(e)}")
                        break
                
                time.sleep(1)
                yield i + len(batch)
        else:
            logger.info("No new documents to add.")

    def get_relevant_documents(self, query: str, initial_top_k: int = 100, final_top_n: int = 15) -> list[str]:
        logger.info(f"Embedding query: '{query}'")
        query_embedding = self.co.embed(
            texts=[query],
            model=self.embed_model,
            input_type="search_query"
        ).embeddings[0]
        logger.info(f"Query embedding shape: {len(query_embedding)}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, embedding FROM documents")
            all_docs = cursor.fetchall()

        doc_ids, contents, embeddings = zip(*all_docs)
        embeddings = [np.frombuffer(emb, dtype=np.float32) for emb in embeddings]
        
        logger.info(f"Number of documents: {len(embeddings)}")
        logger.info(f"Document embedding shape: {embeddings[0].shape}")

        if any(len(emb) != self.embed_dim for emb in embeddings):
            logger.error(f"Embedding dimensions mismatch in database. Expected {self.embed_dim}")
            return []

        logger.info("Calculating similarities...")
        similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))
        top_k_indices = np.argsort(similarities)[-initial_top_k:][::-1]
        top_k_docs = [contents[i] for i in top_k_indices]
        logger.info(f"Found top {initial_top_k} similar documents")

        chunks = []
        for doc in top_k_docs:
            sentences = sent_tokenize(doc)
            chunk_size = 5
            for i in range(0, len(sentences), chunk_size):
                chunk = " ".join(sentences[i:i+chunk_size])
                chunks.append(chunk)

        logger.info(f"Reranking {len(chunks)} chunks...")
        reranked = self.co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=chunks,
            top_n=final_top_n,
            return_documents=True
        )
        logger.info(f"Reranking complete. Got top {final_top_n} chunks.")

        relevant_chunks = [item.document.text for item in reranked.results]
        logger.info(f"Relevant chunks for query '{query}':")
        for chunk in relevant_chunks:
            logger.info(f"- {chunk[:100]}...")
        return relevant_chunks

    def generate_response(self, query: str, context: str) -> str:
        model = st.session_state.llm_model
        max_tokens = st.session_state.max_tokens

        prompt = f"""You are now engaging in a live debate with your opponent for the upcoming presidential election. You are the sitting Democrat President up for reelection; and your opponent is the Republican party's chosen leader: Donald Trump.

        Use the following context, made of speeches and statements of your administration and yourself, to inform your response, but respond in POTUS's voice and style: you are the commander in chief.

        Your response should not contain any disclaimers that this is an impersonation, you answer the question directly Mr. President, your seat and Party's future are at stake.

        Context: {context}
        
        Question: {query}
        
        Response:"""

        messages = [
            {"role": "system", "content": "You are an AI assistant Role-playing the 46th President of the United States, Joe Biden. You will impersonate the republican President Joe Biden at a time where reelection is coming up, and answer as if you were him, in the first person. You do not need to say things like 'here's my response as Potus blabla': answer directly. Don't break the role play."},
            {"role": "user", "content": prompt}
        ]

        logger.info(f"Generating response for query: '{query}'")
        
        try:
            trimmed_messages = trim_messages(messages, model, max_tokens=max_tokens)
            response = completion(
                model=model,
                messages=trimmed_messages,
                api_key=st.session_state.openrouter_api_key
            )
            
            logger.info("Response generated successfully.")
            logger.info(f"Response: {response}")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I'm unable to provide a response at this time due to a technical issue: {str(e)}"

    def process_query(self, query: str) -> str:
        relevant_docs = self.get_relevant_documents(query)
        context = "\n\n".join(relevant_docs)
        return self.generate_response(query, context)


async def text_to_speech_async(text, voice_id, model):
    url = 'https://api.neets.ai/v1/tts'
    headers = {
        'Content-Type': 'application/json',
        'X-API-Key': st.session_state.neets_api_key
    }
    data = {
        'text': text,
        'voice_id': voice_id,
        'params': {
            'model': model
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                return await response.read()
            else:
                st.error(f"TTS API error: {response.status} - {await response.text()}")
                return None

def main():
    st.title("Presidential Debate Chatbot")

    if 'api_keys_entered' not in st.session_state:
        st.session_state.api_keys_entered = False

    if not st.session_state.api_keys_entered:
        st.session_state.cohere_api_key = st.text_input("Enter your Cohere API Key:", type="password")
        st.session_state.neets_api_key = st.text_input("Enter your Neets API Key:", type="password")
        st.session_state.openrouter_api_key = st.text_input("Enter your LLM API Key:", type="password")
        
        model_selection = st.selectbox("Select LLM Model:", LLM_MODELS)
        if model_selection == "custom":
            st.session_state.llm_model = st.text_input("Enter custom model name (format: provider/modelname):")
            st.warning("Please ensure you enter the custom model name in the correct format: provider/modelname")
        else:
            st.session_state.llm_model = model_selection
        
        st.session_state.max_tokens = st.number_input("Set max tokens for input trimming:", min_value=1, max_value=32000, value=15000)
        
        if st.button("Submit API Keys and Settings"):
            st.session_state.api_keys_entered = True
            st.session_state.biden_ai = SQLiteBidenAISystem()
            
            with sqlite3.connect(st.session_state.biden_ai.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents")
                count = cursor.fetchone()[0]
            
            if count == 0:
                document_dir = "/Users/sho/test1/46"
                all_documents = [os.path.join(document_dir, f) for f in os.listdir(document_dir) if f.endswith('.txt')]
                
                documents_content = []
                for doc_path in tqdm(all_documents, desc="Reading documents", unit="doc"):
                    with open(doc_path, 'r', encoding='utf-8') as file:
                        documents_content.append(file.read())
                
                with st.spinner("Generating embeddings and adding documents to the database..."):
                    st.text("This process may take a few minutes. Please wait.")
                    progress_bar = st.progress(0)
                    
                    total_docs = len(documents_content)
                    for i, _ in enumerate(st.session_state.biden_ai.add_documents(documents_content)):
                        progress = (i + 1) / total_docs
                        progress_bar.progress(progress)
                        st.text(f"Processed {i+1}/{total_docs} documents")
                    
                    st.success("Document embedding complete!")
                
                st.success(f"Added {len(documents_content)} documents to the database.")
            else:
                st.info(f"Database already contains {count} documents.")
            
            st.experimental_rerun()

    if st.session_state.api_keys_entered:
        with open('Questions.txt', 'r') as file:
            debate_questions = [q.strip() for q in file.readlines() if q.strip()]

        st.subheader("The Debate Questions:")
        for i, question in enumerate(debate_questions):
            st.write(f"{i+1}. {question}")

        user_input = st.text_input("Ask a question or choose from the debate questions above:")

        question_voice = st.selectbox("Select voice for question:", VOICES, index=VOICES.index('cardi-b'))
        answer_voice = st.selectbox("Select voice for answer:", VOICES, index=VOICES.index('joe-biden'))

        if st.button("Ask"):
            if user_input:
                question_audio = asyncio.run(text_to_speech_async(user_input, question_voice, 'ar-diff-50k'))
                if question_audio:
                    st.audio(question_audio, format='audio/wav')

                response = st.session_state.biden_ai.process_query(user_input)
                st.write("Response:", response)

                response_audio = asyncio.run(text_to_speech_async(response, answer_voice, 'ar-diff-50k'))
                if response_audio:
                    st.audio(response_audio, format='audio/wav')

                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": response})

        st.subheader("Chat History:")
        for message in st.session_state.messages:
            st.write(f"{message['role'].capitalize()}: {message['content']}")

if __name__ == "__main__":
    main()