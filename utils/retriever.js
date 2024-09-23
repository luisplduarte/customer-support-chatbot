import dotenv from 'dotenv';
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from '@supabase/supabase-js'

dotenv.config();

const openAIApiKey = process.env.OPENAI_API_KEY

const embeddings = new OpenAIEmbeddings({ openAIApiKey })

const client = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY)

const retriever = await SupabaseVectorStore.fromExistingIndex(client, embeddings, {
    //client,
    tableName: 'documents',
    queryName: 'match_documents'
})

//const retriever = vectorStore.similaritySearch()

export { retriever }