import dotenv from 'dotenv';
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from '@supabase/supabase-js'

dotenv.config();

/**
 * Function that creates and returns vector database connection
 * @returns database retriever
 */
export const createRetriever = () => {
  // Init Supabase client
  const client = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY);

  // Init embeddings model
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  // Creates and returns Supabase integration object
  return new SupabaseVectorStore(embeddings, {
    client,
    tableName: 'documents',
    queryName: 'match_documents',
  });
};
