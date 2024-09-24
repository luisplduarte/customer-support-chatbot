import dotenv from 'dotenv';
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import supabaseClient from '../supabaseClient.js';

dotenv.config();

/**
 * Function that creates and returns vector database connection
 * @returns database retriever
 */
export const createRetriever = () => {
  // Init embeddings model
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  // Creates and returns Supabase integration object
  return new SupabaseVectorStore(embeddings, {
    client: supabaseClient,
    tableName: 'documents',
    queryName: 'match_documents',
  });
};
