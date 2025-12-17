-- Create profiles table for user information
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT,
  full_name TEXT,
  attempts_used INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON public.profiles FOR INSERT
  WITH CHECK (auth.uid() = id);

-- Create topics table
CREATE TABLE public.topics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  icon_name TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on topics (public read)
ALTER TABLE public.topics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view topics"
  ON public.topics FOR SELECT
  TO authenticated
  USING (true);

-- Create questions table
CREATE TABLE public.questions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  topic_id UUID REFERENCES public.topics(id) ON DELETE CASCADE,
  question_text TEXT NOT NULL,
  ideal_answer TEXT,
  keywords TEXT[], -- Array of keywords for scoring
  difficulty TEXT CHECK (difficulty IN ('Easy', 'Medium', 'Hard')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on questions
ALTER TABLE public.questions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Authenticated users can view questions"
  ON public.questions FOR SELECT
  TO authenticated
  USING (true);

-- Create user_attempts table
CREATE TABLE public.user_attempts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
  selected_topics UUID[] NOT NULL,
  started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at TIMESTAMP WITH TIME ZONE,
  time_taken INTEGER, -- in seconds
  overall_score INTEGER CHECK (overall_score >= 0 AND overall_score <= 100),
  communication_score INTEGER CHECK (communication_score >= 0 AND communication_score <= 100),
  technical_score INTEGER CHECK (technical_score >= 0 AND technical_score <= 100),
  problem_solving_score INTEGER CHECK (problem_solving_score >= 0 AND problem_solving_score <= 100),
  status TEXT DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'abandoned'))
);

-- Enable RLS on user_attempts
ALTER TABLE public.user_attempts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own attempts"
  ON public.user_attempts FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can create own attempts"
  ON public.user_attempts FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own attempts"
  ON public.user_attempts FOR UPDATE
  USING (auth.uid() = user_id);

-- Create attempt_answers table
CREATE TABLE public.attempt_answers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  attempt_id UUID REFERENCES public.user_attempts(id) ON DELETE CASCADE,
  question_id UUID REFERENCES public.questions(id) ON DELETE CASCADE,
  question_text TEXT NOT NULL,
  user_answer TEXT,
  transcript TEXT, -- Voice transcription
  answer_score INTEGER CHECK (answer_score >= 0 AND answer_score <= 100),
  keywords_matched TEXT[],
  feedback TEXT,
  answered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on attempt_answers
ALTER TABLE public.attempt_answers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own answers"
  ON public.attempt_answers FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.user_attempts
      WHERE user_attempts.id = attempt_answers.attempt_id
      AND user_attempts.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create own answers"
  ON public.attempt_answers FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.user_attempts
      WHERE user_attempts.id = attempt_answers.attempt_id
      AND user_attempts.user_id = auth.uid()
    )
  );

-- Create function to auto-create profile
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.email)
  );
  RETURN NEW;
END;
$$;

-- Trigger to create profile on user signup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

-- Add triggers for updated_at
CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_questions_updated_at
  BEFORE UPDATE ON public.questions
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();

-- Insert sample topics
INSERT INTO public.topics (name, description, icon_name) VALUES
  ('Machine Learning', 'AI and ML fundamentals, algorithms, and applications', 'Brain'),
  ('Data Structures', 'Core data structures and their implementations', 'Database'),
  ('System Design', 'Scalable system architecture and design patterns', 'Cpu'),
  ('Algorithms', 'Algorithm analysis, optimization, and problem solving', 'Code'),
  ('Networking', 'Network protocols, architecture, and security', 'Network');

-- Insert sample questions
INSERT INTO public.questions (topic_id, question_text, ideal_answer, keywords, difficulty)
SELECT 
  t.id,
  'Tell me about your experience with machine learning algorithms.',
  'Discussion of supervised/unsupervised learning, common algorithms like regression, classification, neural networks, and practical applications.',
  ARRAY['supervised', 'unsupervised', 'neural networks', 'regression', 'classification', 'training', 'models'],
  'Medium'
FROM public.topics t WHERE t.name = 'Machine Learning';

INSERT INTO public.questions (topic_id, question_text, ideal_answer, keywords, difficulty)
SELECT 
  t.id,
  'How would you design a scalable system for millions of users?',
  'Discussion of load balancing, database sharding, caching strategies, microservices, CDN usage, and horizontal scaling.',
  ARRAY['load balancing', 'scaling', 'caching', 'database', 'microservices', 'distributed', 'performance'],
  'Hard'
FROM public.topics t WHERE t.name = 'System Design';

INSERT INTO public.questions (topic_id, question_text, ideal_answer, keywords, difficulty)
SELECT 
  t.id,
  'Explain the difference between supervised and unsupervised learning.',
  'Supervised learning uses labeled data for training, while unsupervised learning finds patterns in unlabeled data. Examples include classification vs clustering.',
  ARRAY['labeled', 'unlabeled', 'classification', 'clustering', 'training data', 'patterns'],
  'Easy'
FROM public.topics t WHERE t.name = 'Machine Learning';

INSERT INTO public.questions (topic_id, question_text, ideal_answer, keywords, difficulty)
SELECT 
  t.id,
  'What are the key considerations when optimizing database queries?',
  'Indexing strategies, query execution plans, avoiding N+1 queries, proper use of joins, connection pooling, and caching.',
  ARRAY['indexing', 'query optimization', 'joins', 'execution plan', 'caching', 'performance'],
  'Medium'
FROM public.topics t WHERE t.name = 'Data Structures';

INSERT INTO public.questions (topic_id, question_text, ideal_answer, keywords, difficulty)
SELECT 
  t.id,
  'Describe a challenging project you worked on and how you solved it.',
  'Clear problem statement, approach taken, technical challenges faced, solutions implemented, and measurable outcomes.',
  ARRAY['problem solving', 'technical', 'solution', 'implementation', 'results', 'challenges'],
  'Medium'
FROM public.topics t WHERE t.name = 'Algorithms';