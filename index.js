import express from 'express';
import axios from 'axios';
import dotenv from 'dotenv';
import { promises as fs } from 'fs';
import { createClient } from '@supabase/supabase-js';

import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
//import { SupabaseVectorStore } from 'langchain/vectorstores';
import { OpenAI } from "@langchain/openai";

import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from 'langchain/chains';

import { StringOutputParser } from "@langchain/core/output_parsers";
//import { StringOutputParser } from 'langchain/schema/output_parser'
import { ChatPromptTemplate, HumanMessagePromptTemplate } from "@langchain/core/prompts";
import { HumanMessage } from "@langchain/core/messages";

import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables"

import { formatDocumentsAsString } from "langchain/util/document";

import { retriever } from './utils/retriever.js'
import { combineDocuments } from './utils/combineDocuments.js'

import { OpenAIEmbeddings } from "@langchain/openai";
import { OpenAIModerationChain } from "langchain/chains";


dotenv.config();

const app = express();
app.use(express.json());

const openAIApiKey = process.env.OPENAI_API_KEY

const llm = new ChatOpenAI({ openAIApiKey })

const addInitialKnowledgeToSupabase = async () => {
    // Get knowledge from file
    const text = await fs.readFile('knowledge.txt', 'utf-8');
        
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
    const embeddings = new OpenAIEmbeddings({ openAIApiKey });
    const vectors = await embeddings.embedDocuments(documentsWithMetadata.map(doc => doc.pageContent));

    const rows = documentsWithMetadata.map((doc, i) => ({
        content: doc.pageContent,
        embedding: vectors[i],
        metadata: doc.metadata,
    }));

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
        // Turning user input to standalone question

        // A string holding the phrasing of the prompt
        const standaloneQuestionTemplate = 'Given a question, convert it to a standalone question. question: {question} standalone question:'
        const standaloneQuestionPrompt = PromptTemplate.fromTemplate(standaloneQuestionTemplate)
        console.log("\n\n-----------------------------\n standaloneQuestionPrompt = ", standaloneQuestionPrompt)

        const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided. Try to find the answer in the context. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@scrimba.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.
            context: {context}
            question: {question}
            answer:`
        const answerPrompt = PromptTemplate.fromTemplate(answerTemplate)

        // Configuração da LLMChain que combina o PromptTemplate com o modelo
        const standaloneQuestionChain = new LLMChain({
            llm,
            prompt: standaloneQuestionPrompt,
        });
        console.log("standaloneQuestionChain = ", standaloneQuestionChain)

        const answerChain = new LLMChain({
            llm,
            prompt: answerPrompt
        });
        console.log("answerChain = ", answerChain)

        const userQuestion = 'What are the technical requirements for running Scrimba? I only have a very old laptop which is not that powerful.'


        const retrieverChain = RunnableSequence.from([
            async (prevResult) => {
                const standaloneQuestion = prevResult.standalone_question;
                const results = await retriever.similaritySearchWithScore(standaloneQuestion, 1);
                return results;  // Retorna o resultado da busca por similaridade
            },
            combineDocuments
        ])
        console.log("retrieverChain = ", retrieverChain)

        const standaloneResult = await standaloneQuestionChain.call({ question: userQuestion });
        console.log("Standalone Question:", standaloneResult.text);

        /* TODO: it's this chain that is causing problems 
        const chain = RunnableSequence.from([
            {
                standalone_question: await standaloneQuestionChain.call({ question: userQuestion }),
                //original_input: new RunnablePassthrough()
            },
            {
                context: retrieverChain,
                question: ({ original_input }) => original_input.question
            },
            answerChain
        ])

        console.log("chain = ", chain)
        */

        

        const chain = RunnableSequence.from([
            {
              question: (i) => i.question,
              context: async (i) => {
                const relevantDocs = await retriever.similaritySearchWithScore(i.question);
                return formatDocumentsAsString(relevantDocs);
              },
            },
            async (input) => {
              const { question, context } = input;
        
              const messages = [
                {
                  role: "system",
                  content: `You’re an HTML generator. You should generate high-quality HTML based on the image you'll receive.
                  
                   Here are some examples of previously generated HTML and JS code. Base your work primarily on the styles if you find them similar: ${context}.`,
                },
                {
                  role: "user",
                  content: [
                    { type: "text", text: question },
                  ],
                },
              ];
        
              const response = await model.invoke(messages);
        
              return response;
            },
            new StringOutputParser(),
          ]);







        /*
        Await the response when you INVOKE the chain. 
        Remember to pass in a question.
        const response = await standaloneQuestionChain.call({
            question: 'What are the technical requirements for running Scrimba? I only have a very old laptop which is not that powerful.'
        })

        const standaloneQuestionChain = standaloneQuestionPrompt
            .pipe(llm)
            .pipe(new StringOutputParser())
            
        console.log("standaloneQuestionChain = ", standaloneQuestionChain)
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
        
        const response = await chain.invoke({
            question: userQuestion
        })

        console.log(response)
        res.status(200).send(response);
        
        
        /*
        //const chain = standaloneQuestionPrompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriever)

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
