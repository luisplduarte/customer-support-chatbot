import { ChatOpenAI } from "@langchain/openai";

const openAIApiKey = process.env.OPENAI_API_KEY

/**
 * Function that returns the LLM AI model that will be used in the app.
 * @returns The AI model to be used in the app.
 */
export const getModel = async () => {
    const MODEL_OPEN_AI = "open_ai";
  
    const providers = {
      [MODEL_OPEN_AI] : async () => {
          return new ChatOpenAI({ 
            model: "gpt-4o-mini", 
            openAIApiKey 
        })
      },
    }

    return await providers?.[process.env.AI_MODEL]()
  }