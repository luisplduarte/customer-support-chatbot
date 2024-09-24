import dotenv from 'dotenv';
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from '@supabase/supabase-js'

dotenv.config();

const OPEN_AI_API_KEY = process.env.OPENAI_API_KEY
const EMBEDDINGS = new OpenAIEmbeddings({ OPEN_AI_API_KEY })
const CLIENT = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY)

const retriever = await SupabaseVectorStore.fromExistingIndex(EMBEDDINGS, {
    CLIENT,
    tableName: 'documents',
    queryName: 'match_documents'
})

export { retriever }