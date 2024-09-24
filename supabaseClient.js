import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config();

// Init Supabase client
const supabaseClient = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_API_KEY);

export default supabaseClient;
