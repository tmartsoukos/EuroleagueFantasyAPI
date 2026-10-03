-- 002_row_level_security.sql: Row Level Security και περιορισμός δικαιωμάτων.
--
-- Γιατί: το Supabase εκθέτει κάθε πίνακα του schema public στο REST API του (PostgREST) με τους
-- ρόλους anon (δημόσιο κλειδί) και authenticated. Η εφαρμογή (το API στο Render και το ingestion)
-- συνδέεται ΑΠΕΥΘΕΙΑΣ στη βάση με τον ρόλο postgres και δεν χρησιμοποιεί το REST API του Supabase.
-- Άρα κανένας πίνακας δεν πρέπει να είναι προσβάσιμος μέσα από αυτό.
--
-- Πώς:
-- 1. ENABLE ROW LEVEL SECURITY σε κάθε πίνακα, ΧΩΡΙΣ policies: χωρίς policy, ένας ρόλος που
--    υπόκειται σε RLS δεν διαβάζει και δεν αλλάζει καμία γραμμή (deny-all). Ο ρόλος postgres είναι
--    ιδιοκτήτης των πινάκων και έχει το χαρακτηριστικό BYPASSRLS στο Supabase, άρα παρακάμπτει το
--    RLS και η εφαρμογή δουλεύει κανονικά. ΔΕΝ χρησιμοποιείται FORCE ROW LEVEL SECURITY: θα
--    έκλεινε και τον ιδιοκτήτη.
-- 2. Επιπλέον (defense in depth) αφαιρούνται όλα τα δικαιώματα των anon και authenticated στους
--    πίνακες και στην ακολουθία του predictions.id, ώστε να μην υπάρχει πρόσβαση ούτε αν κάποιος
--    απενεργοποιήσει κατά λάθος το RLS. Το βήμα αυτό εκτελείται μόνο αν οι ρόλοι υπάρχουν
--    (pg_roles), ώστε το ίδιο migration να τρέχει και σε απλό Postgres χωρίς Supabase.
--
-- Ο πίνακας schema_migrations δημιουργείται από τον runner (python -m elfantasy.db.migrate), ο
-- οποίος ενεργοποιεί το RLS αμέσως μετά τη δημιουργία. Εδώ το RLS ξαναεπιβεβαιώνεται και για αυτόν,
-- χωρίς σφάλμα αν το αρχείο εφαρμοστεί χωρίς τον runner (τότε ο πίνακας δεν υπάρχει).

ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE players ENABLE ROW LEVEL SECURITY;
ALTER TABLE games ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_games ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_availability ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF to_regclass('schema_migrations') IS NOT NULL THEN
        ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY;
    END IF;
END
$$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        REVOKE ALL ON TABLE teams, players, games, player_games, predictions, player_availability FROM anon;
        REVOKE ALL ON SEQUENCE predictions_id_seq FROM anon;
        IF to_regclass('schema_migrations') IS NOT NULL THEN
            REVOKE ALL ON TABLE schema_migrations FROM anon;
        END IF;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        REVOKE ALL ON TABLE teams, players, games, player_games, predictions, player_availability FROM authenticated;
        REVOKE ALL ON SEQUENCE predictions_id_seq FROM authenticated;
        IF to_regclass('schema_migrations') IS NOT NULL THEN
            REVOKE ALL ON TABLE schema_migrations FROM authenticated;
        END IF;
    END IF;
END
$$;
