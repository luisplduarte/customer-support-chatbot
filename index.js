import express from 'express';
import dotenv from 'dotenv';
import { promises as fs } from 'fs';
import { createClient } from '@supabase/supabase-js';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables"
import { createRetriever } from './utils/retriever.js';
import { OpenAIEmbeddings } from "@langchain/openai";

dotenv.config();

const app = express();
app.use(express.json());

const openAIApiKey = process.env.OPENAI_API_KEY
const LLM_MODEL = new ChatOpenAI({ openAIApiKey })
const retriever = createRetriever();

const addInitialKnowledgeToSupabase = async () => {
    // Get knowledge from file
    const text = await fs.readFile('./knowledge.txt', 'utf-8');
        
    // Split knowledge into smaller chuncks of text
    // This will split the knowledge text into smaller chuncks of text
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        separators: ['\n\n', '\n', ' ', '', '##'], // Default setting to the text is splited respecting the paragraphs and other text separators
        chunkOverlap: 50 //The text will overlap in different chuncks when needed
    })

    const documents = await splitter.createDocuments([text])

    // Add metadata to each document
    const documentsWithMetadata = documents.map((doc, index) => ({
        ...doc,
        metadata: {
            source: 'knowledge.txt',
            chunk: index + 1,
            totalChunks: documents.length
        }
    }));

    // Initialize information embeddings and create vectors
    if (!openAIApiKey) throw new Error(`Expected env var OPENAI_API_KEY`);
    const embeddings = new OpenAIEmbeddings({ openAIApiKey });
    const vectors = await embeddings.embedDocuments(documentsWithMetadata.map(doc => doc.pageContent));

    const rows = documentsWithMetadata.map((doc, i) => ({
        content: doc.pageContent,
        embedding: vectors[i],
        metadata: doc.metadata,
    }));

    //TODO: refactor this createClient
    const client = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY)
    // Store the vectores created into Supabase (vectorial DB)
    const { data, error } = await client.from('documents').insert(rows);

    if (error) {
        throw new Error(`Error inserting: ${error.message}`);
    }
}

app.post('/init', async (req, res) => {
    try {
        addInitialKnowledgeToSupabase()

        res.status(200).send('Documents added to Supabase');

      } catch (err) {
        console.log(err)
        res.status(500).send('Internal Server Error');
      }
});

app.post('/chat', async (req, res) => {
    try {
        //const userQuestion = 'Who is the best football player?'
        const userQuestion = 'What is Scrimba?'
        console.log("userQuestion = ", userQuestion)
        
        // A string holding the phrasing of the prompt
        const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. question: {question} standalone question:'
        const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)

        const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
            context: {context}
            question: {question}
            answer:`
        const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

        //TODO: add validations

        // Gets top 3 closest vectores/results from DB
        const retrieverChain = RunnableSequence.from([
            async (prevResult) => {
                //console.log("\n\n standaloneQuestion = ", prevResult)
                const results = await retriever.similaritySearch(prevResult, 3);
                //console.log("\n\n results = ", results)
                return results;
            },
            //TODO: change these documents transformations
            (docs) => docs.flat().map((doc) => doc.pageContent).join('\n\n\n\n')
        ])
        //console.log("retrieverChain = ", retrieverChain)

        // Turning user input to standalone question
        const standaloneQuestion = await standaloneQuestionPrompt.pipe(LLM_MODEL).invoke({ question: userQuestion });

        //TODO: merge 2 runnable
        const chain = RunnableSequence.from([            
            async (prevRes) => {
              return retrieverChain.invoke(prevRes.question.content);
            },
            async (docs) => {
              //console.log("docs = ", docs)
              return {
                context: docs,
                question: userQuestion
              }
            },
            answerPrompt.pipe(LLM_MODEL)
        ])

        //console.log("chain = ", chain)

        const response = await chain.invoke({
            question: standaloneQuestion
        })
        console.log("\n\n AI response = ", response.content)

        
        /*
        Await the response when you INVOKE the chain. 
        Remember to pass in a question.
        const response = await standaloneQuestionChain.call({
            question: 'What are the technical requirements for running Scrimba? I only have a very old laptop which is not that powerful.'
        })

        const standaloneQuestionChain = standaloneQuestionPrompt
            .pipe(LLM_MODEL)
            .pipe(new StringOutputParser())
            
        */
        
        
        // Runnable sequence that executes the process of generating a standalone question,
        // performing a similarity search, and returning the answer
        /*
        const chain = RunnableSequence.from([
            async (input) => {
                // Step 1: Generate a standalone question
                const standaloneResult = await standaloneQuestionChain.call({ question: userQuestion });
                console.log("\n\n-----------------------------\n standaloneResult = ", standaloneResult)
                const standaloneQuestion = standaloneResult.text.trim();
                console.log("\n\n-----------------------------\n standaloneQuestion = ", standaloneQuestion)
        
                // Step 2: Retrieve context documents based on similarity
                const retrieverResult = await retriever.similaritySearchWithScore(standaloneQuestion, 1);
                console.log("\n\n-----------------------------\n retrieverResult = ", retrieverResult)
        
                // Step 3: Combine documents for final context
                const combinedDocs = await combineDocuments(retrieverResult);
                console.log("\n\n-----------------------------\n combinedDocs = ", combinedDocs)
        
                // Step 4: Generate the final answer
                const answerResult = await answerChain.call({
                    context: combinedDocs,
                    question: userQuestion
                });
                console.log("\n\n-----------------------------\n Answer Result:", answerResult.text);
        
                //return answerResult.text;
            }
        ]);
        */
        
        /*
        //const chain = standaloneQuestionPrompt.pipe(LLM_MODEL).pipe(new StringOutputParser()).pipe(retriever)

        const response = await chain.invoke({
            question: 'What are the technical requirements for running Scrimba? I only have a very old laptop which is not that powerful.'
        })

        console.log(response)
        res.status(200).send(response);
        */

        res.status(200).send();
    } catch (err) {
        console.log(err)
        res.status(500).send('Internal Server Error');
    }
});


app.listen(3000, () => {
  console.log('Chatbot server running on port 3000');
});
