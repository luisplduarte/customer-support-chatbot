import dotenv from 'dotenv';
import { OpenAIEmbeddings } from "@langchain/openai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient } from '@supabase/supabase-js';
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";

dotenv.config();

// Init embeddings model
const openAIApiKey = process.env.OPENAI_API_KEY
const embeddings = new OpenAIEmbeddings({ openAIApiKey });  

// Init Supabase client
const supabaseClient = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY);

// Init Pinecone client
const pinecone = new PineconeClient();
const pineconeClient = pinecone.Index(process.env.PINECONE_INDEX);  // Will automatically read the PINECONE_API_KEY and PINECONE_ENVIRONMENT env vars

/**
 * Function that creates and returns vector database connection
 * @returns database retriever
 */
export const createRetriever = async () => {
  //TODO: refactor this providers array to a configuration file
  const PROVIDER_SUPABASE = "supabase";
  const PROVIDER_PINECONE = "pinecone";

  const providers = {
    [PROVIDER_SUPABASE] : async () => {
        return new SupabaseVectorStore(embeddings, {
          client: supabaseClient,
          tableName: 'documents',
          queryName: 'match_documents',
        });
    },

    [PROVIDER_PINECONE] : async () => {
        return await PineconeStore.fromExistingIndex(embeddings, {
          pineconeIndex: pineconeClient,
          maxConcurrency: 3,  // Maximum number of batch requests to allow at once. Each batch is 1000 vectors
        });
    }
  }
  return await providers?.[process.env.VECTOR_DB]()
}

/**
 * Creates vectores from knowledge and inserts them into DB
 * @param {array} documents knowledge chunks formatted as Documents type
 * @returns data from DB insertion
 */
export const addInitialKnowledgeToVectorStore = async (documents) => {
  const vectorDb = process.env.VECTOR_DB;

  const docsContents = documents.map(doc => doc.pageContent)

  // Create vectors from arg documents
  const vectors = await embeddings.embedDocuments(docsContents);

  if (vectorDb === 'supabase') {
    // Vector formatting
    const rows = documents.map((doc, i) => ({
        content: doc.pageContent,
        embedding: vectors[i],
        metadata: doc.metadata,
    }));
    
    // Store the vectores created into Supabase (vectorial DB)
    const { data, error } = await supabaseClient.from('documents').insert(rows);
    if (error) throw new Error(`Error inserting: ${error.message}`);

    return data;
  } else if (vectorDb === 'pinecone') {
    //TODO: Change this createRetriever call from here
    const retriever = await createRetriever();
    // In this case we don't need to create the embeddings, pinecone will do it by himself
    return await retriever.addDocuments( documents );
  } else {
    throw new Error('Invalid VECTOR_DB value. Must be either "supabase" or "pinecone".')
  }
}
