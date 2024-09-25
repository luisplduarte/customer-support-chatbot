import express from 'express';
import dotenv from 'dotenv';
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables"
import { createRetriever, addInitialKnowledgeToSupabase } from './utils/database.js';
import { combineDocuments, getKnowledge, knowledgeFormat } from './utils/helpers.js';
import { STANDALONE_TEMPLATE, ANSWER_TEMPLATE } from './utils/constants.js';

dotenv.config();

const app = express();
app.use(express.json());

const openAIApiKey = process.env.OPENAI_API_KEY
const LLM_MODEL = new ChatOpenAI({ openAIApiKey })
const retriever = createRetriever();

app.post('/init', async (req, res) => {
    try {
        if (!openAIApiKey) throw new Error(`Expected env var OPENAI_API_KEY`);

        const file_path = `${import.meta.dirname}/knowledge.txt`
        const knowledge = await getKnowledge(file_path)
        const documentsWithMetadata = await knowledgeFormat(file_path, knowledge)

        addInitialKnowledgeToSupabase(documentsWithMetadata)

        res.status(200).send('Documents added to Supabase');
      } catch (err) {
        console.log(err)
        res.status(500).send('Internal Server Error');
      }
});

const conversationHitory = [];

//TODO: find a way to not use the history array
app.post('/chat', async (req, res) => {
    try {
        const { userQuestion } = req.body
        if(!userQuestion) throw new Error(`Missing user question!`);

        //TODO: simplify the code by creating more functions

        // Append current user question to history
        conversationHitory.push({ role: 'user', content: userQuestion }); // History structure -> [{ role: "user", content: "message" }, { role: "bot", content: "response" }, ...]
        const formattedConversationHistory = conversationHitory.map(h => `${h.role}: ${h.content}`).join('\n');
        
        // Standalone question prompt holding the string phrasing of the standalone prompt
        const standaloneQuestionPrompt = PromptTemplate.fromTemplate(STANDALONE_TEMPLATE);
        // Turning user input to standalone question
        const standaloneQuestion = await standaloneQuestionPrompt
          .pipe(LLM_MODEL)
          .invoke({ 
            question: userQuestion,
            conversation_history: formattedConversationHistory
          });

        // Answer prompt holding the string phrasing of the response prompt
        const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE)

        // In this chain we will merge the original user input, standalone question, knowledge from DB and AI response to create the final response
        const chain = RunnableSequence.from([            
            async (prevRes) => await retriever.similaritySearch(prevRes.question.content, 3), // Retrieve top 3 closest vectores/results from DB based on similarity
            (docs) => combineDocuments(docs),  // Combine documents for final context. "docs" is the output of the previous code (the search result in the retriever)
            async (docs) => {
              return {
                context: docs,  // The context is the documents (knowledge) queried from the DB
                question: userQuestion,  // We pass the original user question again because it could still contain relevant information (user sentiment, question context, etc.)
                conversation_history: [...formattedConversationHistory, docs], // Combine the history context and the retrieved knowledge into a single context
              }
            },
            answerPrompt.pipe(LLM_MODEL)  // Merge the answer prompt with the llm model knowledge, the knowledge from DB and original user question
        ]);

        // Generate final answer
        const response = await chain.invoke({
            question: standaloneQuestion
        });

        // Append AI response to history
        conversationHitory.push({ role: 'bot', content: response.content });

        res.status(200).send({ response: response.content, history: conversationHitory });
    } catch (err) {
        console.log(err);
        res.status(500).send('Internal Server Error');
    }
});

app.listen(3000, () => {
  console.log('Chatbot server running on port 3000');
});
