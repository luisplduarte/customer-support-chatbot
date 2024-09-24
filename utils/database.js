import dotenv from 'dotenv';
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from '@supabase/supabase-js';

dotenv.config();

// Init embeddings model
const openAIApiKey = process.env.OPENAI_API_KEY
const embeddings = new OpenAIEmbeddings({ openAIApiKey });  

// Init Supabase client
const supabaseClient = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY);

/**
 * Function that creates and returns vector database connection
 * @returns database retriever
 */
export const createRetriever = () => {
  // Creates and returns Supabase integration object
  return new SupabaseVectorStore(embeddings, {
    client: supabaseClient,
    tableName: 'documents',
    queryName: 'match_documents',
  });
};

/**
 * Creates vectores from knowledge and inserts them into DB
 * @param {*} documents knowledge chuncks formatted as Documents type
 * @returns data from DB insertion
 */
export const addInitialKnowledgeToSupabase = async (documents) => {
  // Create vectors from arg documents
  const vectors = await embeddings.embedDocuments(documents.map(doc => doc.pageContent));

  // Vector formatting
  const rows = documents.map((doc, i) => ({
      content: doc.pageContent,
      embedding: vectors[i],
      metadata: doc.metadata,
  }));
  
  // Store the vectores created into Supabase (vectorial DB)
  const { data, error } = await supabaseClient.from('documents').insert(rows);
  if (error) throw new Error(`Error inserting: ${error.message}`);

  return data
}
