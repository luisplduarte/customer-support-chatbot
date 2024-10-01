import express from 'express';
import dotenv from 'dotenv';
import { createClient as createRedisClient } from 'redis';
import { addInitialKnowledgeToVectorStore } from './utils/database.js';
import { getKnowledge, knowledgeFormat } from './utils/helpers.js';
import { generateResponse } from './responseGenerator.js';

dotenv.config();

const app = express();
app.use(express.json());

const redisClient = createRedisClient({
  url: process.env.REDIS_URL,  // Certifique-se de adicionar REDIS_URL no arquivo .env
});
await redisClient.connect();

app.post('/init', async (req, res) => {
    try {
        if (!process.env.OPENAI_API_KEY) throw new Error(`Expected env var OPENAI_API_KEY`);

        const file_path = `${import.meta.dirname}/knowledge.txt`
        const knowledge = await getKnowledge(file_path)
        const documentsWithMetadata = await knowledgeFormat(file_path, knowledge)

        addInitialKnowledgeToVectorStore(documentsWithMetadata)

        res.status(200).send('Documents added to Supabase');
      } catch (err) {
        console.log(err)
        res.status(500).send('Internal Server Error');
      }
});

app.post('/chat', async (req, res) => {
    try {
        const { userQuestion, history, conversationId } = req.body
        if(!userQuestion) throw new Error(`Missing user question!`);
        if(!history) throw new Error(`Missing history!`);

        let chatHistory = history;
        if (conversationId) {
            const redisHistory = await redisClient.get(conversationId);
            if (redisHistory) {
                chatHistory = JSON.parse(redisHistory);
            }
        } else {
            const newConversationId = `conversation_${Date.now()}`;
            chatHistory = [{ role: 'user', content: userQuestion }];
            await redisClient.setEx(newConversationId, 3600, JSON.stringify(chatHistory));  // Expires in 1 hour
        }

        //history.push({ role: 'user', content: userQuestion }); // Append current user question to history
        const ai_response = await generateResponse(userQuestion, history)
        //history.push({ role: 'bot', content: ai_response });  // Append AI response to history
        chatHistory.push({ role: 'bot', content: ai_response });  // Append AI response to history

        // Update history
        await redisClient.setEx(conversationId || `conversation_${Date.now()}`, 3600, JSON.stringify(chatHistory));

        res.status(200).send({ response: ai_response, history: history });
    } catch (err) {
        console.log(err);
        res.status(500).send('Internal Server Error');
    }
});

app.listen(3000, () => {
  console.log('Chatbot server running on port 3000');
});
