-- Create filter_states table for saving/loading filter configurations
CREATE TABLE IF NOT EXISTS filter_states (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    filter_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries by user_id
CREATE INDEX IF NOT EXISTS idx_filter_states_user_id ON filter_states(user_id);

-- Create index for ordering by creation date
CREATE INDEX IF NOT EXISTS idx_filter_states_created_at ON filter_states(created_at DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE filter_states ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to only access their own filter states
CREATE POLICY "Users can only access their own filter states" ON filter_states
    FOR ALL USING (auth.uid() = user_id);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_filter_states_updated_at
    BEFORE UPDATE ON filter_states
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();