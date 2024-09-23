
const express = require('express');
const axios = require('axios');
const OpenAI = require('openai');
require('dotenv').config();

const app = express();
app.use(express.json());

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

//TODO: change this to read file
// Sample knowledge base (you could also use a database)
const knowledgeBase = {
  "What is your refund policy?": "We offer a 30-day money-back guarantee on all purchases.",
  "How do I track my order?": "You can track your order using the tracking number sent to your email."
};

// Function to check own knowledge base
const checkKnowledgeBase = (query) => {
  for (let key in knowledgeBase) {
    if (query.toLowerCase().includes(key.toLowerCase())) {
      return knowledgeBase[key];
    }
  }
  return null; // If no match is found
};

// OpenAI API call function
const getAIResponse = async (query) => {
  try {
    const response = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [{ role: 'user', content: query }],
      max_tokens: 150,
    });
    return response.choices[0].message.content.trim();
  } catch (error) {
    console.error('Error with OpenAI API:', error);
    return "Sorry, I'm unable to process that request at the moment.";
  }
};

app.post('/chat', async (req, res) => {
  const { message } = req.body;

  // Check if the question is in our own knowledge base
  const knowledgeBaseResponse = checkKnowledgeBase(message);
  if (knowledgeBaseResponse) {
    return res.json({ response: knowledgeBaseResponse });
  }

  // If not in knowledge base, query OpenAI API
  const aiResponse = await getAIResponse(message);
  res.json({ response: aiResponse });
});

app.listen(3000, () => {
  console.log('Chatbot server running on port 3000');
});

const RecursiveCharacterTextSplitter = require('langchain/text_splitter')

try {
  const result = await fetch('knowledge.txt')
  const text = await result.text()

  //This will split the knowledge text into smaller chuncks of text
  const splitter = new RecursiveCharacterTextSplitter()
  
  const output = await splitter.createDocuments([text])
  console.log(output)

} catch (err) {
  console.log(err)
}

