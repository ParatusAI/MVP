-- schema.sql
-- Drop tables if they exist (for fresh start)
DROP TABLE IF EXISTS material_properties;
DROP TABLE IF EXISTS spectral_data;
DROP TABLE IF EXISTS synthesis_runs;

-- Create synthesis_runs table
CREATE TABLE synthesis_runs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    cs_flow_rate FLOAT NOT NULL,
    pb_flow_rate FLOAT NOT NULL,
    temperature FLOAT NOT NULL,
    residence_time FLOAT NOT NULL,
    status VARCHAR(20) DEFAULT 'running',
    notes TEXT
);

-- Create spectral_data table
CREATE TABLE spectral_data (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES synthesis_runs(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT NOW(),
    wavelength FLOAT NOT NULL,
    intensity FLOAT NOT NULL,
    measurement_type VARCHAR(10) NOT NULL CHECK (measurement_type IN ('uv_vis', 'pl', 'ftir'))
);

-- Create material_properties table
CREATE TABLE material_properties (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES synthesis_runs(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT NOW(),
    plqy FLOAT,
    emission_peak FLOAT,
    fwhm FLOAT,
    particle_size FLOAT,
    bandgap FLOAT,
    stability_metric FLOAT
);

-- Create indexes for better performance
CREATE INDEX idx_spectral_data_run_id ON spectral_data(run_id);
CREATE INDEX idx_spectral_data_measurement_type ON spectral_data(measurement_type);
CREATE INDEX idx_material_properties_run_id ON material_properties(run_id);
CREATE INDEX idx_synthesis_runs_timestamp ON synthesis_runs(timestamp);