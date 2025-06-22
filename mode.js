// const fs = require('fs');
import fs from 'fs'
import { pipeline } from '@xenova/transformers';
import axios from 'axios'
// const axios = require('axios');

async function getEmbeddings(chunks) {
  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  const vectors = [];
  for (const chunk of chunks) {
    const embedding = await embedder(chunk, { pooling: 'mean', normalize: true });
    vectors.push({ chunk, embedding });
  }
  return vectors;
}

function cosineSimilarity(a, b) {
    const dot = a.data.reduce((sum, val, i) => sum + val * b.data[i], 0);
    const normA = Math.sqrt(a.data.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.data.reduce((sum, val) => sum + val * val, 0));
    return dot / (normA * normB);
  }
  
function retrieveTopChunks(queryEmbedding, db, k = 5) {
    return db
      .map(entry => ({
        chunk: entry.chunk,
        similarity: cosineSimilarity(queryEmbedding, entry.embedding),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k)
      .map(e => e.chunk);
  }
  

function loadConversationChunks(path) {
  const data = JSON.parse(fs.readFileSync(path, 'utf-8'));
  const messages = data.map(
    m => `${m.id.fromMe ? 'Person1' : 'Person2'}: ${m.body}`
  );
  return messages.join('\n');
}


function chunkText(text, chunkSize = 1000, overlap = 100) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize - overlap) {
      chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
  }

async function askLLaMA3(prompt, apiKey) {
    const res = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: "meta-llama/llama-3-8b-instruct",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: prompt }
        ]
      },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        }
      }
    );
    return res.data.choices[0].message.content.trim();
  }

 (async () => {
    const apiKey = "";
    const conversationText = loadConversationChunks("_.json");
    const chunks = chunkText(conversationText);
  
    const embeddedChunks = await getEmbeddings(chunks);
  
    const question = "Do you think Person 2 likes person 1";
    const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    const questionEmbedding = await embedder(question, { pooling: 'mean', normalize: true });
  
    const topChunks = retrieveTopChunks(questionEmbedding, embeddedChunks, 5);
    const context = topChunks.join("\n\n");
    const finalPrompt = `Based on the following conversation excerpts:\n\n${context}\n\nAnswer the question:\n${question}`;
  
    const answer = await askLLaMA3(finalPrompt, apiKey);
    console.log("Answer:", answer);
  })();
  
