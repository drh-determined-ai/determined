DELETE FROM projects
WHERE id = $1
AND (user_id = $2 OR $3 IS TRUE)
RETURNING projects.id;
