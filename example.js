"@langchain/community": "^0.2.31",
"@langchain/openai": "^0.2.8"


import { ChatOpenAI } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";

import { TextLoader } from "langchain/document_loaders/fs/text";

import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient, SupabaseClient } from "@supabase/supabase-js";

import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";

import "dotenv/config";

const REPO_PATH = "./src/external_knowledge";

const loadExternalKnowledgeFromDocuments = async (repoPath: string) => {
  const loader = new DirectoryLoader(repoPath, {
    ".js": (path) => new TextLoader(path),
    ".html": (path) => new TextLoader(path),
  });
  const docs = await loader.load();

  const javascriptSplitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
    chunkSize: 2000,
    chunkOverlap: 200,
  });
  const texts = await javascriptSplitter.splitDocuments(docs);

  console.log("Loaded ", texts.length, " documents.");

  return texts;
};

const createSupabaseClient = () => {
  const privateKey = process.env.SUPABASE_PRIVATE_KEY;
  if (!privateKey) throw new Error(`Expected env var SUPABASE_PRIVATE_KEY`);

  const url = process.env.SUPABASE_URL;
  if (!url) throw new Error(`Expected env var SUPABASE_URL`);

  const client = createClient(url, privateKey);

  return client;
};

const createVectorStoreFromDocuments = async (
  texts: Document<Record<string, any>>[],
  client: SupabaseClient<any, "public", any>,
  apiKey: string
) => {
  const vectorStore = await SupabaseVectorStore.fromDocuments(
    texts,
    new OpenAIEmbeddings({ apiKey }),
    {
      client,
      tableName: "documents",
      queryName: "match_documents",
    }
  );

  return vectorStore;
};

const setupLLM = (apiKey: string, llmModel: "OpenAI" | "Claude") => {
  const modelsMap = {
    OpenAI: new ChatOpenAI({ model: "gpt-4o-mini", apiKey }).pipe(
      new StringOutputParser()
    ),
    Claude: new ChatAnthropic({
      model: "claude-3-5-sonnet-20240620",
      apiKey,
    }),
  };

  return modelsMap[llmModel];
};

const promptLLM = async (
  vectorStore: SupabaseVectorStore,
  question: string,
  imageBase64: string,
  llmModel: "OpenAI" | "Claude",
  apiKey: string
) => {
  const retriever = vectorStore.asRetriever({
    searchType: "mmr", // Use max marginal relevance search
    searchKwargs: { fetchK: 3 },
  });

  const model = setupLLM(apiKey, llmModel);

  const conversationalQaChain = RunnableSequence.from([
    {
      question: (i: { question: string; image: string }) => i.question,
      image: (i: { question: string; image: string }) => i.image,
      context: async (i: { question: string; image: string }) => {
        const relevantDocs = await retriever.invoke(i.question);
        return formatDocumentsAsString(relevantDocs);
      },
    },
    async (input) => {
      const { question, image, context } = input;

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
            { type: "image_url", image_url: { url: `${image}` } },
          ],
        },
      ];

      const response = await model.invoke(messages);

      return response;
    },
    new StringOutputParser(),
  ]);

  const result = await conversationalQaChain.invoke({
    question,
    image: imageBase64,
  });

  console.log("Result:", result);
  return result;
};

let cachedTexts: Document<Record<string, any>>[] | null = null;
let cachedClient: SupabaseClient<any, "public", any>;
let cachedExternalKnowledgeVectorStore: SupabaseVectorStore | null = null;

const generateCode = async (
  imageBase64: string,
  llmModel: "OpenAI" | "Claude",
  apiKey: string
) => {
  try {
    if (!cachedTexts) {
      cachedTexts = await loadExternalKnowledgeFromDocuments(REPO_PATH);
    }

    if (!cachedClient) {
      cachedClient = createSupabaseClient();
    }

    if (!cachedExternalKnowledgeVectorStore) {
      cachedExternalKnowledgeVectorStore = await createVectorStoreFromDocuments(
        cachedTexts,
        cachedClient,
        apiKey
      );
    }

    const result = await promptLLM(
      cachedExternalKnowledgeVectorStore,
      " Generate the HTML email layout represented in this image, respecting its styles. In your response, please output only the HTML without any markdown annotations.",
      imageBase64,
      llmModel,
      apiKey
    );

    return {
      success: true,
      result,
    };
  } catch (error) {
    console.error("Error generating code:", error);
    return {
      success: false,
      data: `Error: while generating code`,
    };
  }
};

export { generateCode };