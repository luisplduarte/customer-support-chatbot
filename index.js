import express from 'express';
import session from 'express-session';
import dotenv from 'dotenv';
import { createClient as createRedisClient } from 'redis';
import RedisStore from "connect-redis"
import { addInitialKnowledgeToVectorStore } from './utils/database.js';
import { getKnowledge, knowledgeFormat } from './utils/helpers.js';
import { generateResponse } from './responseGenerator.js';

dotenv.config();

const app = express();
app.use(express.json());

const redisClient = createRedisClient();
await redisClient.connect().catch(console.error);

// Redis connection store configuration
const redisStore = new RedisStore({
  client: redisClient
});

// Redis sessions configuration
app.use(
  session({
    store: redisStore,
    secret: process.env.SESSION_SECRET || 'mySecret',
    resave: false,
    saveUninitialized: false,
    cookie: { maxAge: 1000 * 60 * 60 }, // 1 hour
  })
);

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
        const { userQuestion } = req.body
        if(!userQuestion) throw new Error(`Missing user question!`);

        const history = req.session.history || [];
        history.push({ role: 'user', content: userQuestion }); // Append current user question to history

        const ai_response = await generateResponse(userQuestion, history)

        history.push({ role: 'bot', content: ai_response });  // Append AI response to history
        req.session.history = history;

        res.status(200).send({ 
          response: ai_response, 
          history: history
        });
    } catch (err) {
        console.log(err);
        res.status(500).send('Internal Server Error');
    }
});

app.listen(3000, () => {
  console.log('Chatbot server running on port 3000');
});
