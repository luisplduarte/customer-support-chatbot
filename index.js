import express from 'express';
import dotenv from 'dotenv';
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables"
import { createRetriever, addInitialKnowledgeToSupabase } from './utils/database.js';
import { combineDocuments, getKnowledge, knowledgeFormat } from './utils/helpers.js';

dotenv.config();

const app = express();
app.use(express.json());

const openAIApiKey = process.env.OPENAI_API_KEY
const LLM_MODEL = new ChatOpenAI({ openAIApiKey })
const retriever = createRetriever();

const STANDALONE_TEMPLATE = 'Given a question, convert it to a standalone question. question: {question} standalone question:'
const ANSWER_TEMPLATE = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that.", and direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
  context: {context}
  question: {question}
  answer:`

app.post('/init', async (req, res) => {
    try {
        if (!openAIApiKey) throw new Error(`Expected env var OPENAI_API_KEY`);

        //TODO: get knowledge path from request ???
        const knowledge = await getKnowledge('./knowledge.txt')
        const documentsWithMetadata = await knowledgeFormat('./knowledge.txt', knowledge)

        addInitialKnowledgeToSupabase(documentsWithMetadata)

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
        
        // Standalone question prompt holding the string phrasing of the standalone prompt
        const standaloneQuestionPrompt = PromptTemplate.fromTemplate(STANDALONE_TEMPLATE)
        // Turning user input to standalone question
        const standaloneQuestion = await standaloneQuestionPrompt.pipe(LLM_MODEL).invoke({ question: userQuestion });

        // Answer prompt holding the string phrasing of the response prompt
        const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE)

        // In this chain we will merge the original user input, standalone question, knowledge from DB and AI response to create the final response
        const chain = RunnableSequence.from([            
            async (prevRes) => await retriever.similaritySearch(prevRes.question.content, 3), // Retrieve top 3 closest vectores/results from DB based on similarity
            (docs) => combineDocuments(docs),  // Combine documents for final context. "docs" is the output of the previous code (the search result in the retriever)
            async (docs) => {
              return {
                context: docs,  // The context is the documents (knowledge) queried from the DB
                question: userQuestion  // We pass the original user question again because it could still contain relevant information (user sentiment, question context, etc.)
              }
            },
            answerPrompt.pipe(LLM_MODEL)  // Merge the answer prompt with the llm model knowledge, the knowledge from DB and original user question
        ])

        // Generate final answer
        const response = await chain.invoke({
            question: standaloneQuestion
        })

        res.status(200).send(response.content);
    } catch (err) {
        console.log(err)
        res.status(500).send('Internal Server Error');
    }
});

app.listen(3000, () => {
  console.log('Chatbot server running on port 3000');
});
