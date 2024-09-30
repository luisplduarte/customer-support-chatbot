import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { promises as fs } from 'fs';

/**
 * Function to combine the documents into one
 * @param {*} docs Documents to be combined
 * @returns String with document's page content combined 
 */
export const combineDocuments = (docs) => {
    return docs.flat().map((doc) => doc.pageContent).join('\n\n\n\n')
}

/**
 * This function reads a .txt file to get the knowledge.
 * @param {*} filePath File's path.
 * @returns String with the file content.
 */
export const getKnowledge = (filePath) => {
    return fs.readFile(filePath, 'utf-8');
}

/**
 * This function transforms the knowledge that is passed as string into documents
 * @param {string} filePath the path of the file where knowledge is located
 * @param {string} knowledge string with knowledge to be formatted
 * @returns Document array with formatted knowledge
 */
export const knowledgeFormat = async (filePath, knowledge) => {
    // Text splitter configuration
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        separators: ['\n\n', '\n', ' ', '', '##'], // Text is splited respecting the paragraphs and other text separators
        chunkOverlap: 50 // The text will overlap in different chuncks when needed
    })

    // This will split the knowledge into smaller chuncks of text
    const documents = await splitter.createDocuments([knowledge])

    // Add metadata to each document
    return documents.map((doc, index) => ({
        ...doc,
        metadata: {
            source: filePath,
            chunk: index + 1,
            totalChunks: documents.length
        }
    }));
}
