-- =========================================================
-- TrustLens AI - Supabase (PostgreSQL) Schema & Data Migration
-- =========================================================

-- 1. Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    userid VARCHAR(50) UNIQUE NOT NULL,
    fullname VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Migrate existing user accounts from MySQL
INSERT INTO users (id, userid, fullname, email, password, created_at)
VALUES 
(1, 'pavisuba015', 'PAVITHRA', 'pavithramkp127@gmail.com', 'scrypt:32768:8:1$zI1jnw7Sv5RYeOk6$18702072297456160292420b045eec980ac79eae686facade06b9dbc897f8ff677ae9c854c232cd044e792038addb27b95d1941559102ecfa031844e71a60f82', '2026-07-16 21:05:18+00'),
(2, 'bhuvana07', 'BHUVANA', 'bhuvanapalani399@gmail.com', 'scrypt:32768:8:1$gjq7MRQDet8YgHsL$30af616404d736b600b3cca6b866a9d8d53149537d07cef637dc02024843cbebb14a0345b72cff461f824b5497e5ef6877bf0fadcc9ef9cc39678fd7a1213cd7', '2026-07-17 14:43:18+00'),
(3, '0001', 'sathiya', '2211ysathiya@gmail.com', 'scrypt:32768:8:1$s7xTI4zqBrk0Jy8P$38997fbd5903394d52ee5e3e6ed2dc9b02430e12ac68233b7a0fa5bc58206280056cf9c0229e1e2d57f7ccc6388a6646d05d787728c423ef6605d3026cad5d3f', '2026-08-13 13:37:52+00')
ON CONFLICT (userid) DO NOTHING;

-- 3. Sync the serial sequence to ensure new signups work with auto-increment IDs
SELECT setval(pg_get_serial_sequence('users', 'id'), COALESCE(MAX(id), 1)) FROM users;
