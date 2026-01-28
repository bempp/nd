// Simple triangle mesh with parametric coordinates
// This geo file creates a surface without boundary elements

SetFactory("OpenCASCADE");

// Create a simple rectangle (avoid circle boundary curve issues)
Rectangle(1) = {0, 0, 0, 1, 1};

// Generate a coarse mesh
Mesh.CharacteristicLengthMin = 0.4;
Mesh.CharacteristicLengthMax = 0.4;

// Enable parametric coordinates in output
Mesh.SaveParametric = 1;

// Only save surface elements (no point/curve elements)
Mesh.SaveAll = 0;
Physical Surface(1) = {1};
