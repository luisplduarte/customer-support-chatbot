import express from 'express';
import dotenv from 'dotenv';
import { promises as fs } from 'fs';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables"
import { createRetriever } from './utils/retriever.js';
import { OpenAIEmbeddings } from "@langchain/openai";
import supabaseClient from './supabaseClient.js';
import combineDocuments from './utils/combineDocuments.js'

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

/**
 * This function reads a .txt file to get the knowledge
 */
const getKnowledge = (filePath) => {
  return fs.readFile(filePath, 'utf-8');
}

/**
 * This function transforms the knowledge that is passed as string into documents
 * @param {string} filePath the path of the file where knowledge is located
 * @param {string} knowledge string with knowledge to be formatted
 * @returns Document array with formatted knowledge
 */
const knowledgeFormat = async (filePath, knowledge) => {
  // Text splitter configuration
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    separators: ['\n\n', '\n', ' ', '', '##'], // Text is splited respecting the paragraphs and other text separators
    chunkOverlap: 50 // The text will overlap in different chuncks when needed
  })

  // This will split the knowledge into smaller chuncks of text
  const documents = await splitter.createDocuments([knowledge])

  // Add metadata to each document
  return documents.map((doc, index) => ({
    ...doc,
    metadata: {
        source: filePath,
        chunk: index + 1,
        totalChunks: documents.length
    }
  }));
}

/**
 * Creates vectores from knowledge and inserts them into DB
 * @param {*} documents knowledge chuncks formatted as Documents type
 * @returns data from DB insertion
 */
const addInitialKnowledgeToSupabase = async (documents) => {
    // Initialize information embeddings and create vectors
    const embeddings = new OpenAIEmbeddings({ openAIApiKey });
    const vectors = await embeddings.embedDocuments(documents.map(doc => doc.pageContent));

    // Vector formatting
    const rows = documents.map((doc, i) => ({
        content: doc.pageContent,
        embedding: vectors[i],
        metadata: doc.metadata,
    }));
    
    // Store the vectores created into Supabase (vectorial DB)
    const { data, error } = await supabaseClient.from('documents').insert(rows);

    if (error) {
        throw new Error(`Error inserting: ${error.message}`);
    }

    return data
}

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

//TODO: add validations

app.post('/chat', async (req, res) => {
    try {
        const { userQuestion } = req.body
        
        // Standalone question prompt holding the string phrasing of the standalone prompt
        const standaloneQuestionPrompt = PromptTemplate.fromTemplate(STANDALONE_TEMPLATE)

        // Answer prompt holding the string phrasing of the response prompt
        const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE)

        // Retrieve top 3 closest vectores/results from DB based on similarity
        const retrieverChain = RunnableSequence.from([
            async (prevResult) => await retriever.similaritySearch(prevResult, 3),   // prevResult is the arg that we passed in the invoke function
            // docs is the output of the previous code (the search result in the retriever)
            (docs) => combineDocuments(docs)  // Combine documents for final context
        ])

        //TODO: merge 2 runnable
        const chain = RunnableSequence.from([            
            async (prevRes) => retrieverChain.invoke(prevRes.question.content),
            async (docs) => {
              return {
                context: docs,  // The context is the documents (knowledge) queried from the DB
                question: userQuestion  // We pass the original user question again because it could still contain relevant information (user sentiment, question context, etc.)
              }
            },
            answerPrompt.pipe(LLM_MODEL)  // Merge the answer prompt with the llm model knowledge, the knowledge from DB and original user question
        ])

        // Turning user input to standalone question
        const standaloneQuestion = await standaloneQuestionPrompt.pipe(LLM_MODEL).invoke({ question: userQuestion });

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
