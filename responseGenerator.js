import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables"
import { combineDocuments } from './utils/helpers.js';
import { STANDALONE_TEMPLATE, ANSWER_TEMPLATE } from './utils/constants.js';
import { createRetriever } from './utils/database.js';
import { getModel } from "./utils/aiModels.js";

const LLM_MODEL = await getModel();
const retriever = await createRetriever();

/**
 * This function receives user question and chat history and returns the response created by the AI model.
 * It uses user question, chat history, standalone question, knowledge from DB and answer prompt so the LLM model generates
 * a response related to the question.
 * @param {string} userQuestion User question
 * @param {array} history History array with questions and answers from user and chatbot. History structure = [{ role: "user", content: "message" }, { role: "bot", content: "response" }, ...]
 * @returns response created by the AI model.
 */
export const generateResponse = async (userQuestion, history) => {
    const formattedConversationHistory = history.map(h => `${h.role}: ${h.content}`).join('\n');
    
    // Turning user input to standalone question
    const standaloneQuestionPrompt = PromptTemplate.fromTemplate(STANDALONE_TEMPLATE);
    const standaloneQuestion = await standaloneQuestionPrompt
        .pipe(LLM_MODEL)
        .invoke({ 
            question: userQuestion,
            conversation_history: formattedConversationHistory
        });

    // Answer prompt holding the string phrasing of the response prompt
    const answerPrompt = PromptTemplate.fromTemplate(ANSWER_TEMPLATE)

    // In this chain we will merge the original user input, standalone question, knowledge from DB and AI response to create the final response
    const chain = RunnableSequence.from([            
        async (prevRes) => await retriever.similaritySearch(prevRes.question.content, 3), // Retrieve top 3 closest vectores/results from DB based on similarity
        (docs) => combineDocuments(docs),  // Combine documents for final context. "docs" is the output of the previous code (the search result in the retriever)
        async (docs) => {
            return {
            context: docs,  // The context is the documents (knowledge) queried from the DB
            question: userQuestion,  // We pass the original user question again because it could still contain relevant information (user sentiment, question context, etc.)
            conversation_history: [...formattedConversationHistory, docs], // Combine the history context and the retrieved knowledge into a single context
            }
        },
        answerPrompt.pipe(LLM_MODEL)  // Merge previous info sources with answer prompt and llm model knowledge
    ]);

    // Generate final answer
    const response = await chain.invoke({
        question: standaloneQuestion
    });

    return response.content;
}