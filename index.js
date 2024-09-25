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
const LLM_MODEL = new ChatOpenAI({ 
  openAIApiKey, 
  temperature: 0 // Set temperature to 0 cecause this is a chatbot and we don't want him to be creative in the response
})
const retriever = createRetriever();

// This standalone template will tell AI to convert user question to standalone question (simplifies the text) as well as 
//give the conversation history context to the prompt
const STANDALONE_TEMPLATE = `Given some conversation history (if any) and a question, convert the question to a standalone question. 
  conversation history: {conversation_history}
  question: {question} 
  standalone question: `

// This answer template will tell AI how to respont to user question. We give the context (knowledge), conversation history
//as well as the user question to the prompt so it has more information to create the response
const ANSWER_TEMPLATE = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided and the conversation history provided. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible and reply using you own words. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
  context: {context}
  conversation history: {conversation_history}
  question: {question}
  answer: `

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

const conversationHitory = [];

app.post('/chat', async (req, res) => {
    try {
        const { userQuestion } = req.body
        if(!userQuestion) throw new Error(`Missing user question!`);

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
