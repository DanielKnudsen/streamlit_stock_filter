-- SQL to run in Supabase SQL Editor:

-- 1. Add frequency column to existing filter_states table
ALTER TABLE filter_states ADD COLUMN IF NOT EXISTS frequency TEXT DEFAULT 'never';

-- 2. Create email_notifications table for tracking email sends
CREATE TABLE IF NOT EXISTS email_notifications (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    filter_id UUID REFERENCES filter_states(id),
    sent_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT CHECK (status IN ('sent', 'failed', 'bounced')),
    recipient_email TEXT,
    frequency TEXT,
    results_count INTEGER
);

-- 3. Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_email_notifications_user_id_sent_at ON email_notifications(user_id, sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_email_notifications_status ON email_notifications(status);
