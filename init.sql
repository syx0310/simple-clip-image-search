CREATE EXTENSION vector;

CREATE TABLE md5_path (
    id SERIAL PRIMARY KEY,
    md5_hash VARCHAR(32) NOT NULL,
    file_path VARCHAR(255) NOT NULL
);

CREATE TABLE test (
	id VARCHAR, 
	embedding vector(768), 
	embtype int ,
	CONSTRAINT pk_test PRIMARY KEY (id, embtype)
);