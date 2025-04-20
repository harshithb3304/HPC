#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define NUM_NODES 100
#define NUM_FEATURES 20
#define NUM_EPOCHS 5
#define LEARNING_RATE 0.001
#define GRADIENT_CLIP 1.0f

// Define the node structure with neighbors
typedef struct
{
    int *neighbors;
    int num_neighbors;
} Node;

// Parameters for the GNN model
typedef struct
{
    float **attention_weights; // (NUM_FEATURES x NUM_FEATURES)
    float *attention_bias;     // (NUM_FEATURES)
    float **weights;           // (NUM_FEATURES x NUM_FEATURES)
    float *node_bias;          // (NUM_FEATURES) - Added node bias
} GNNParameters;

// Structure to hold training labels
typedef struct
{
    int *node_pairs; // (NUM_TRAIN_PAIRS x 2)
    float *labels;   // (NUM_TRAIN_PAIRS)
    int num_pairs;
} TrainingData;

// -------------------- Activation and Norm Functions --------------------

// ReLU activation
void relu(float *features, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (features[i] < 0)
            features[i] = 0;
    }
}

// Sigmoid activation
float sigmoid(float x)
{
    // Clip x to avoid overflow
    float clipped_x = fmaxf(-20.0f, fminf(x, 20.0f));
    return 1.0f / (1.0f + expf(-clipped_x));
}

// Layer normalization with scale and shift parameters
void layer_norm(float *features, int size)
{
    float mean = 0.0f;
    float variance = 0.0f;

    // Calculate mean
    for (int i = 0; i < size; i++)
    {
        mean += features[i];
    }
    mean /= size;

    // Calculate variance
    for (int i = 0; i < size; i++)
    {
        variance += (features[i] - mean) * (features[i] - mean);
    }
    variance /= size;

    // Normalize with epsilon for stability
    float std_dev = sqrtf(variance + 1e-8f);
    for (int i = 0; i < size; i++)
    {
        features[i] = (features[i] - mean) / std_dev;
    }
}

// -------------------- Memory Management Functions --------------------

// Allocate 2D matrix
float **allocate_matrix(int rows, int cols)
{
    float **matrix = (float **)malloc(rows * sizeof(float *));
    if (!matrix)
    {
        perror("Memory allocation failed for matrix rows");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
        if (!matrix[i])
        {
            perror("Memory allocation failed for matrix columns");
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

// Free 2D matrix
void free_matrix(float **matrix, int rows)
{
    if (matrix)
    {
        for (int i = 0; i < rows; i++)
        {
            if (matrix[i])
            {
                free(matrix[i]);
            }
        }
        free(matrix);
    }
}

// -------------------- File I/O Functions --------------------

// Read a matrix from a file (elements stored row-wise)
float **read_matrix(const char *filename, int rows, int cols)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening matrix file");
        // If file doesn't exist, return a randomly initialized matrix
        float **matrix = allocate_matrix(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i][j] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f; // Initialize between -0.1 and 0.1
            }
        }
        printf("Initialized random matrix for %s\n", filename);
        return matrix;
    }

    float **matrix = allocate_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (fscanf(file, "%f", &matrix[i][j]) != 1)
            {
                fprintf(stderr, "Error reading matrix element [%d][%d] from %s\n", i, j, filename);
                // Fill remaining with random values
                matrix[i][j] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
            }
        }
    }
    fclose(file);
    return matrix;
}

// Read a vector from a file
float *read_vector(const char *filename, int size)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Error opening vector file");
        // If file doesn't exist, return a randomly initialized vector
        float *vector = (float *)malloc(size * sizeof(float));
        for (int i = 0; i < size; i++)
        {
            vector[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f; // Initialize between -0.1 and 0.1
        }
        printf("Initialized random vector for %s\n", filename);
        return vector;
    }

    float *vector = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        if (fscanf(file, "%f", &vector[i]) != 1)
        {
            fprintf(stderr, "Error reading vector element [%d] from %s\n", i, filename);
            // Fill remaining with random values
            vector[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
        }
    }
    fclose(file);
    return vector;
}

// Save model parameters to files
void save_parameters(const GNNParameters *params)
{
    FILE *file;

    // Save attention weights
    file = fopen("attention_weights.txt", "w");
    if (file)
    {
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                fprintf(file, "%f ", params->attention_weights[i][j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }

    // Save weights
    file = fopen("weights.txt", "w");
    if (file)
    {
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                fprintf(file, "%f ", params->weights[i][j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }

    // Save attention bias
    file = fopen("attention_bias.txt", "w");
    if (file)
    {
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            fprintf(file, "%f\n", params->attention_bias[i]);
        }
        fclose(file);
    }

    // Save node bias
    file = fopen("node_bias.txt", "w");
    if (file)
    {
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            fprintf(file, "%f\n", params->node_bias[i]);
        }
        fclose(file);
    }
}

// Initialize parameters from files or randomly if files don't exist
void initialize_parameters(GNNParameters *params)
{
    params->attention_weights = read_matrix("attention_weights.txt", NUM_FEATURES, NUM_FEATURES);
    params->weights = read_matrix("weights.txt", NUM_FEATURES, NUM_FEATURES);
    params->attention_bias = read_vector("attention_bias.txt", NUM_FEATURES);
    params->node_bias = read_vector("node_bias.txt", NUM_FEATURES);
}

// Allocate and zero-initialize gradients
void initialize_gradients(GNNParameters *gradients)
{
    gradients->attention_weights = allocate_matrix(NUM_FEATURES, NUM_FEATURES);
    gradients->weights = allocate_matrix(NUM_FEATURES, NUM_FEATURES);
    gradients->attention_bias = (float *)calloc(NUM_FEATURES, sizeof(float));
    gradients->node_bias = (float *)calloc(NUM_FEATURES, sizeof(float));

    // Zero-initialize all gradients
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            gradients->attention_weights[i][j] = 0.0f;
            gradients->weights[i][j] = 0.0f;
        }
        gradients->attention_bias[i] = 0.0f;
        gradients->node_bias[i] = 0.0f;
    }
}

// -------------------- GNN Functions --------------------

// Compute attention score between two nodes with scaling
float attention_score(const float *node_i, const float *node_j, const GNNParameters *params)
{
    float score = 0.0f;

    // More efficient matrix multiplication for attention score
    for (int k = 0; k < NUM_FEATURES; k++)
    {
        float temp = params->attention_bias[k] * node_i[k];
        for (int l = 0; l < NUM_FEATURES; l++)
        {
            temp += node_i[k] * params->attention_weights[k][l] * node_j[l];
        }
        score += temp;
    }

    // Scale by sqrt(feature_dim) as in "Attention Is All You Need"
    return score / sqrtf(NUM_FEATURES);
}

// Softmax function for attention scores with numerical stability
void softmax(float *scores, int size)
{
    if (size <= 0)
        return;

    // Find maximum value for numerical stability
    float max_score = scores[0];
    for (int i = 1; i < size; i++)
    {
        if (scores[i] > max_score)
            max_score = scores[i];
    }

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        // Clip to prevent overflow
        scores[i] = expf(fminf(scores[i] - max_score, 20.0f));
        sum += scores[i];
    }

    // Prevent division by zero
    if (sum < 1e-9f)
        sum = 1e-9f;

    // Normalize
    for (int i = 0; i < size; i++)
    {
        scores[i] /= sum;
    }
}

// Forward pass: GNN layer with multi-head attention and skip connections
void gnn_layer(const Node *graph, float **node_features, const GNNParameters *params, float ***attention_cache)
{
    // Create temporary storage for updated features
    float **updated_features = allocate_matrix(NUM_NODES, NUM_FEATURES);

    for (int i = 0; i < NUM_NODES; i++)
    {
        // Copy original features for skip connection
        memcpy(updated_features[i], node_features[i], NUM_FEATURES * sizeof(float));

        if (graph[i].num_neighbors > 0)
        {
            // Allocate memory for attention scores
            float *scores = (float *)malloc(graph[i].num_neighbors * sizeof(float));
            float *aggregated = (float *)calloc(NUM_FEATURES, sizeof(float));

            // Calculate attention scores for each neighbor
            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                int neighbor = graph[i].neighbors[j];
                scores[j] = attention_score(node_features[i], node_features[neighbor], params);
            }

            // Apply softmax to get attention weights
            softmax(scores, graph[i].num_neighbors);

            // Cache attention scores for backward pass
            attention_cache[i] = (float **)malloc(2 * sizeof(float *));
            attention_cache[i][0] = scores;
            attention_cache[i][1] = (float *)malloc(graph[i].num_neighbors * sizeof(float));
            memcpy(attention_cache[i][1], scores, graph[i].num_neighbors * sizeof(float));

            // Weighted sum of neighbor features
            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                int neighbor = graph[i].neighbors[j];
                for (int k = 0; k < NUM_FEATURES; k++)
                {
                    aggregated[k] += scores[j] * node_features[neighbor][k];
                }
            }

            // Apply linear transformation
            for (int k = 0; k < NUM_FEATURES; k++)
            {
                float transformed = params->node_bias[k];
                for (int l = 0; l < NUM_FEATURES; l++)
                {
                    transformed += aggregated[l] * params->weights[k][l];
                }
                updated_features[i][k] += transformed;
            }

            free(aggregated);
        }

        // Apply layer normalization
        layer_norm(updated_features[i], NUM_FEATURES);

        // Apply ReLU activation
        relu(updated_features[i], NUM_FEATURES);
    }

    // Update node features
    for (int i = 0; i < NUM_NODES; i++)
    {
        memcpy(node_features[i], updated_features[i], NUM_FEATURES * sizeof(float));
    }

    // Free temporary storage
    free_matrix(updated_features, NUM_NODES);
}

// Binary cross-entropy loss
float binary_cross_entropy(float pred, float target)
{
    // Strong bounds to ensure valid log inputs
    const float epsilon = 1e-7f;
    pred = fmaxf(epsilon, fminf(pred, 1.0f - epsilon));

    // Compute loss and check for NaN
    float loss = -target * logf(pred) - (1.0f - target) * logf(1.0f - pred);
    if (isnan(loss))
    {
        printf("Warning: NaN loss detected! pred=%f, target=%f\n", pred, target);
        return 0.0f; // Return safe value instead of propagating NaN
    }
    return loss;
}

// Calculate loss and gradients for link prediction
float calculate_loss_and_gradients(float **node_features, const TrainingData *training_data,
                                   float **feature_gradients)
{
    float total_loss = 0.0f;
    for (int i = 0; i < NUM_NODES; i++)
    {
        memset(feature_gradients[i], 0, NUM_FEATURES * sizeof(float));
    }
    for (int p = 0; p < training_data->num_pairs; p++)
    {
        int node1 = training_data->node_pairs[p * 2];
        int node2 = training_data->node_pairs[p * 2 + 1];
        float target = training_data->labels[p];
        float similarity = 0.0f;
        for (int k = 0; k < NUM_FEATURES; k++)
        {
            similarity += node_features[node1][k] * node_features[node2][k];
        }

        float prediction = sigmoid(similarity);

        float loss = binary_cross_entropy(prediction, target);
        total_loss += loss;

        float d_loss = (prediction - target) / (prediction * (1.0f - prediction) + 1e-9f);

        for (int k = 0; k < NUM_FEATURES; k++)
        {
            feature_gradients[node1][k] += d_loss * node_features[node2][k];
            feature_gradients[node2][k] += d_loss * node_features[node1][k];
        }
    }

    return total_loss / training_data->num_pairs;
}

// Backward pass: Compute gradients for model parameters
void backward_pass(const Node *graph, float **node_features, const GNNParameters *params,
                   GNNParameters *gradients, float ***attention_cache, float **feature_gradients)
{
    // Reset gradients to zero
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            gradients->weights[i][j] = 0.0f;
            gradients->attention_weights[i][j] = 0.0f;
        }
        gradients->attention_bias[i] = 0.0f;
        gradients->node_bias[i] = 0.0f;
    }

    // Temporary storage for gradients through layer norm and ReLU
    float **normalized_gradients = allocate_matrix(NUM_NODES, NUM_FEATURES);

    // Apply ReLU and layer norm gradients
    for (int i = 0; i < NUM_NODES; i++)
    {
        // Copy initial gradients
        memcpy(normalized_gradients[i], feature_gradients[i], NUM_FEATURES * sizeof(float));

        // ReLU gradient
        for (int k = 0; k < NUM_FEATURES; k++)
        {
            if (node_features[i][k] <= 0)
            {
                normalized_gradients[i][k] = 0.0f;
            }
        }

        // Layer norm gradient (simplified)
        float sum_grad = 0.0f;
        for (int k = 0; k < NUM_FEATURES; k++)
        {
            sum_grad += normalized_gradients[i][k];
        }
        float mean_grad = sum_grad / NUM_FEATURES;

        // Adjust gradients for layer normalization effect
        for (int k = 0; k < NUM_FEATURES; k++)
        {
            normalized_gradients[i][k] -= mean_grad / NUM_FEATURES;
        }
    }

    // Process each node's contribution to gradients
    for (int i = 0; i < NUM_NODES; i++)
    {
        if (graph[i].num_neighbors > 0 && attention_cache[i] != NULL)
        {
            float *attention_scores = attention_cache[i][1];

            // Gradients for node bias
            for (int k = 0; k < NUM_FEATURES; k++)
            {
                gradients->node_bias[k] += normalized_gradients[i][k];
            }

            // Gradients for weight matrix
            for (int k = 0; k < NUM_FEATURES; k++)
            {
                for (int l = 0; l < NUM_FEATURES; l++)
                {
                    float grad_contribution = 0.0f;
                    for (int j = 0; j < graph[i].num_neighbors; j++)
                    {
                        int neighbor = graph[i].neighbors[j];
                        grad_contribution += attention_scores[j] * node_features[neighbor][l] * normalized_gradients[i][k];
                    }
                    gradients->weights[k][l] += grad_contribution;
                }
            }

            // Calculate attention score gradients
            float *attention_grad = (float *)calloc(graph[i].num_neighbors, sizeof(float));
            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                int neighbor = graph[i].neighbors[j];
                for (int k = 0; k < NUM_FEATURES; k++)
                {
                    float transformed_grad = 0.0f;
                    for (int l = 0; l < NUM_FEATURES; l++)
                    {
                        transformed_grad += params->weights[l][k] * normalized_gradients[i][l];
                    }
                    attention_grad[j] += transformed_grad * node_features[neighbor][k];
                }
            }

            // Apply softmax gradient
            float sum_attention_contrib = 0.0f;
            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                sum_attention_contrib += attention_grad[j] * attention_scores[j];
            }

            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                attention_grad[j] = attention_scores[j] * (attention_grad[j] - sum_attention_contrib);
            }

            // Gradients for attention weights and bias
            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                int neighbor = graph[i].neighbors[j];
                float score_grad = attention_grad[j] / sqrtf(NUM_FEATURES);

                for (int k = 0; k < NUM_FEATURES; k++)
                {
                    gradients->attention_bias[k] += score_grad * node_features[i][k];

                    for (int l = 0; l < NUM_FEATURES; l++)
                    {
                        gradients->attention_weights[k][l] += score_grad * node_features[i][k] * node_features[neighbor][l];
                    }
                }
            }

            free(attention_grad);
        }
    }

    // Free temporary gradient storage
    free_matrix(normalized_gradients, NUM_NODES);
}

// Gradient clipping to prevent exploding gradients
void clip_gradients_mpi(GNNParameters *gradients, int rank, int size)
{
    int total_elements = NUM_FEATURES * NUM_FEATURES;

    // First, create flat arrays for MPI operations
    float *flat_weights = (float *)malloc(total_elements * sizeof(float));
    float *flat_attention_weights = (float *)malloc(total_elements * sizeof(float));

    // Convert 2D matrices to flat arrays
    int idx = 0;
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            flat_weights[idx] = gradients->weights[i][j];
            flat_attention_weights[idx] = gradients->attention_weights[i][j];
            idx++;
        }
    }

    // Calculate local work size
    int elements_per_process = total_elements / size;
    int start_idx = rank * elements_per_process;
    int end_idx = (rank == size - 1) ? total_elements : start_idx + elements_per_process;

    // Perform clipping on local portion of the matrices
    for (int i = start_idx; i < end_idx; i++)
    {
        flat_weights[i] = fmaxf(fminf(flat_weights[i], GRADIENT_CLIP), -GRADIENT_CLIP);
        flat_attention_weights[i] = fmaxf(fminf(flat_attention_weights[i], GRADIENT_CLIP), -GRADIENT_CLIP);
    }

    // Gather results back to all processes
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  flat_weights, elements_per_process, MPI_FLOAT,
                  MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  flat_attention_weights, elements_per_process, MPI_FLOAT,
                  MPI_COMM_WORLD);

    // Handle the bias vectors (divide work among processes)
    int bias_elements_per_process = NUM_FEATURES / size;
    int bias_start = rank * bias_elements_per_process;
    int bias_end = (rank == size - 1) ? NUM_FEATURES : bias_start + bias_elements_per_process;

    for (int i = bias_start; i < bias_end; i++)
    {
        gradients->attention_bias[i] = fmaxf(fminf(gradients->attention_bias[i], GRADIENT_CLIP), -GRADIENT_CLIP);
        gradients->node_bias[i] = fmaxf(fminf(gradients->node_bias[i], GRADIENT_CLIP), -GRADIENT_CLIP);
    }

    // Synchronize bias vectors across processes
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  gradients->attention_bias, bias_elements_per_process, MPI_FLOAT,
                  MPI_COMM_WORLD);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  gradients->node_bias, bias_elements_per_process, MPI_FLOAT,
                  MPI_COMM_WORLD);

    // Convert flat arrays back to 2D matrices
    idx = 0;
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            gradients->weights[i][j] = flat_weights[idx];
            gradients->attention_weights[i][j] = flat_attention_weights[idx];
            idx++;
        }
    }

    // Free temporary arrays
    free(flat_weights);
    free(flat_attention_weights);
}

// Optimized MPI version of gradient clipping (using Scatterv/Gatherv for uneven distribution)
void clip_gradients_mpi_optimized(GNNParameters *gradients, int rank, int size)
{
    // Calculate total elements in weights matrices
    int total_matrix_elements = NUM_FEATURES * NUM_FEATURES;

    // Flatten the matrices for easier MPI operations
    float *flat_weights = NULL;
    float *flat_attention_weights = NULL;
    float *local_weights = NULL;
    float *local_attention_weights = NULL;

    // Arrays for scatter/gather operations
    int *sendcounts = NULL;
    int *displs = NULL;
    int local_count;

    // Calculate local work distribution
    int base_count = total_matrix_elements / size;
    int remainder = total_matrix_elements % size;

    // Each process calculates its local count
    local_count = (rank < remainder) ? base_count + 1 : base_count;

    // Allocate memory for local portions
    local_weights = (float *)malloc(local_count * sizeof(float));
    local_attention_weights = (float *)malloc(local_count * sizeof(float));

    // Prepare sendcounts and displs on all processes
    sendcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    // Calculate send counts and displacements for scatter operation
    for (int i = 0, disp = 0; i < size; i++)
    {
        sendcounts[i] = (i < remainder) ? base_count + 1 : base_count;
        displs[i] = disp;
        disp += sendcounts[i];
    }

    // Rank 0 prepares the flattened data
    if (rank == 0)
    {
        flat_weights = (float *)malloc(total_matrix_elements * sizeof(float));
        flat_attention_weights = (float *)malloc(total_matrix_elements * sizeof(float));

        // Flatten the matrices
        int idx = 0;
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                flat_weights[idx] = gradients->weights[i][j];
                flat_attention_weights[idx] = gradients->attention_weights[i][j];
                idx++;
            }
        }
    }

    // Scatter the data to all processes
    MPI_Scatterv(flat_weights, sendcounts, displs, MPI_FLOAT,
                 local_weights, local_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(flat_attention_weights, sendcounts, displs, MPI_FLOAT,
                 local_attention_weights, local_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Each process clips its local portion
    for (int i = 0; i < local_count; i++)
    {
        local_weights[i] = fmaxf(fminf(local_weights[i], GRADIENT_CLIP), -GRADIENT_CLIP);
        local_attention_weights[i] = fmaxf(fminf(local_attention_weights[i], GRADIENT_CLIP), -GRADIENT_CLIP);
    }

    // Gather the results back
    MPI_Gatherv(local_weights, local_count, MPI_FLOAT,
                flat_weights, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_attention_weights, local_count, MPI_FLOAT,
                flat_attention_weights, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Handle bias vectors with similar logic for load balancing
    int total_bias = NUM_FEATURES;
    int bias_base_count = total_bias / size;
    int bias_remainder = total_bias % size;
    int local_bias_count = (rank < bias_remainder) ? bias_base_count + 1 : bias_base_count;

    // Reallocate sendcounts and displs for bias vectors
    for (int i = 0, disp = 0; i < size; i++)
    {
        sendcounts[i] = (i < bias_remainder) ? bias_base_count + 1 : bias_base_count;
        displs[i] = disp;
        disp += sendcounts[i];
    }

    // Allocate local bias arrays
    float *local_attention_bias = (float *)malloc(local_bias_count * sizeof(float));
    float *local_node_bias = (float *)malloc(local_bias_count * sizeof(float));

    // Scatter bias vectors
    MPI_Scatterv(gradients->attention_bias, sendcounts, displs, MPI_FLOAT,
                 local_attention_bias, local_bias_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(gradients->node_bias, sendcounts, displs, MPI_FLOAT,
                 local_node_bias, local_bias_count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Clip local bias vectors
    for (int i = 0; i < local_bias_count; i++)
    {
        local_attention_bias[i] = fmaxf(fminf(local_attention_bias[i], GRADIENT_CLIP), -GRADIENT_CLIP);
        local_node_bias[i] = fmaxf(fminf(local_node_bias[i], GRADIENT_CLIP), -GRADIENT_CLIP);
    }

    // Gather bias vectors
    MPI_Gatherv(local_attention_bias, local_bias_count, MPI_FLOAT,
                gradients->attention_bias, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_node_bias, local_bias_count, MPI_FLOAT,
                gradients->node_bias, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Rank 0 unpacks the flattened matrices
    if (rank == 0)
    {
        int idx = 0;
        for (int i = 0; i < NUM_FEATURES; i++)
        {
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                gradients->weights[i][j] = flat_weights[idx];
                gradients->attention_weights[i][j] = flat_attention_weights[idx];
                idx++;
            }
        }

        // Free the flattened arrays
        free(flat_weights);
        free(flat_attention_weights);
    }

    // Free local memory
    free(local_weights);
    free(local_attention_weights);
    free(local_attention_bias);
    free(local_node_bias);
    free(sendcounts);
    free(displs);

    // Make sure all processes have updated gradients
    MPI_Barrier(MPI_COMM_WORLD);
}

// Original sequential clip_gradients function (kept for reference/testing)
void clip_gradients(GNNParameters *gradients)
{
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            gradients->weights[i][j] = fmaxf(fminf(gradients->weights[i][j], GRADIENT_CLIP), -GRADIENT_CLIP);
            gradients->attention_weights[i][j] = fmaxf(fminf(gradients->attention_weights[i][j], GRADIENT_CLIP), -GRADIENT_CLIP);
        }
        gradients->attention_bias[i] = fmaxf(fminf(gradients->attention_bias[i], GRADIENT_CLIP), -GRADIENT_CLIP);
        gradients->node_bias[i] = fmaxf(fminf(gradients->node_bias[i], GRADIENT_CLIP), -GRADIENT_CLIP);
    }
}

// Update parameters using gradients with adaptive learning rate
void update_parameters(GNNParameters *params, const GNNParameters *gradients, float learning_rate)
{
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            params->weights[i][j] -= learning_rate * gradients->weights[i][j];
            params->attention_weights[i][j] -= learning_rate * gradients->attention_weights[i][j];
        }
        params->attention_bias[i] -= learning_rate * gradients->attention_bias[i];
        params->node_bias[i] -= learning_rate * gradients->node_bias[i];
    }
}

// Modified prediction function with debugging
float predict_link(int node1, int node2, float **node_features)
{
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (int k = 0; k < NUM_FEATURES; k++)
    {
        dot_product += node_features[node1][k] * node_features[node2][k];
        norm1 += node_features[node1][k] * node_features[node1][k];
        norm2 += node_features[node2][k] * node_features[node2][k];
    }

    // Debug extremely small norms
    if (norm1 < 1e-6f || norm2 < 1e-6f)
    {
        printf("Warning: Very small norm detected in prediction: norm1=%e, norm2=%e\n", norm1, norm2);
    }

    // Very strong protection against division by zero
    float epsilon = 1e-10f;
    norm1 = fmaxf(epsilon, norm1);
    norm2 = fmaxf(epsilon, norm2);

    float cosine_similarity = dot_product / (sqrtf(norm1) * sqrtf(norm2));

    // Ensure similarity is in valid range
    cosine_similarity = fmaxf(-1.0f, fminf(cosine_similarity, 1.0f));
    return (cosine_similarity + 1.0f) / 2.0f; // Scale to [0,1]
}

// Generate training data for link prediction
TrainingData *generate_training_data(const Node *graph, int num_nodes, float pos_neg_ratio)
{
    int max_pairs = num_nodes * 10; // Adjust based on your needs
    TrainingData *training_data = (TrainingData *)malloc(sizeof(TrainingData));

    training_data->node_pairs = (int *)malloc(2 * max_pairs * sizeof(int));
    training_data->labels = (float *)malloc(max_pairs * sizeof(float));

    int pair_count = 0;

    // First add positive examples (actual edges)
    for (int i = 0; i < num_nodes && pair_count < max_pairs; i++)
    {
        for (int j = 0; j < graph[i].num_neighbors && pair_count < max_pairs; j++)
        {
            int neighbor = graph[i].neighbors[j];

            // Avoid duplicates and self-loops
            if (i < neighbor)
            {
                training_data->node_pairs[pair_count * 2] = i;
                training_data->node_pairs[pair_count * 2 + 1] = neighbor;
                training_data->labels[pair_count] = 1.0f;
                pair_count++;
            }
        }
    }

    // Calculate how many negative examples we need
    int positive_count = pair_count;
    int negative_count = (int)(positive_count / pos_neg_ratio);

    // Add negative examples (non-edges)
    int attempts = 0;
    while (pair_count < positive_count + negative_count && attempts < max_pairs * 10)
    {
        int i = rand() % num_nodes;
        int j = rand() % num_nodes;
        attempts++;

        // Avoid self-loops
        if (i == j)
            continue;

        // Check if this is actually a non-edge
        int is_edge = 0;
        for (int k = 0; k < graph[i].num_neighbors; k++)
        {
            if (graph[i].neighbors[k] == j)
            {
                is_edge = 1;
                break;
            }
        }

        if (!is_edge)
        {
            training_data->node_pairs[pair_count * 2] = i;
            training_data->node_pairs[pair_count * 2 + 1] = j;
            training_data->labels[pair_count] = 0.0f;
            pair_count++;
        }
    }

    training_data->num_pairs = pair_count;
    printf("Generated %d training pairs (%d positive, %d negative)\n",
           pair_count, positive_count, pair_count - positive_count);

    return training_data;
}

// -------------------- Main Function --------------------
int main(int argc, char **argv)
{
    int rank, size;
    // Set random seed for reproducibility
    srand(42);

    // Allow parameterized thread count
    int thread_counts[] = {1};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Open file to write benchmark results
    FILE *fp = fopen("benchmark_results.txt", "w");
    if (!fp)
    {
        perror("Error opening benchmark_results.txt");
        return 1;
    }

    fprintf(fp, "Threads,Time(s),Loss,AUC\n");

    for (int test = 0; test < num_tests; test++)
    {
        int num_threads = thread_counts[test];
        omp_set_num_threads(num_threads);

        // printf("\n===== Running test with %d threads =====\n", num_threads);

        // Initialize graph with realistic structure
        Node *graph = (Node *)malloc(NUM_NODES * sizeof(Node));
        for (int i = 0; i < NUM_NODES; i++)
        {
            // More realistic connection pattern (power law distribution)
            graph[i].num_neighbors = 1 + (int)(10.0f * powf((float)rand() / RAND_MAX, 2.0f));
            graph[i].neighbors = (int *)malloc(graph[i].num_neighbors * sizeof(int));

            for (int j = 0; j < graph[i].num_neighbors; j++)
            {
                // Prefer connections to closer indices (model community structure)
                float preference = (float)rand() / RAND_MAX;
                int offset = (int)(NUM_NODES * powf(preference, 2.0f));
                graph[i].neighbors[j] = (i + offset) % NUM_NODES;
            }
        }

        // Initialize node features with better distribution
        float **node_features = allocate_matrix(NUM_NODES, NUM_FEATURES);
        for (int i = 0; i < NUM_NODES; i++)
        {
            for (int j = 0; j < NUM_FEATURES; j++)
            {
                // Normal-ish distribution centered at 0
                float sum = 0;
                for (int k = 0; k < 12; k++)
                {
                    sum += (float)rand() / RAND_MAX;
                }
                node_features[i][j] = (sum - 6.0f) / 3.0f;
            }
            // Normalize features
            layer_norm(node_features[i], NUM_FEATURES);
        }

        // Initialize model parameters
        GNNParameters params, gradients;
        initialize_parameters(&params);
        initialize_gradients(&gradients);

        // Generate training data
        TrainingData *training_data = generate_training_data(graph, NUM_NODES, 1.0f);

        // Allocate cache for attention scores
        float ***attention_cache = (float ***)malloc(NUM_NODES * sizeof(float **));
        for (int i = 0; i < NUM_NODES; i++)
        {
            attention_cache[i] = NULL;
        }

        // Allocate feature gradients
        float **feature_gradients = allocate_matrix(NUM_NODES, NUM_FEATURES);

        // Training loop with timing
        double start_time = omp_get_wtime();
        float adaptive_lr = LEARNING_RATE;

        printf("Starting training for %d epochs\n", NUM_EPOCHS);
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
        {
            // Forward pass
            gnn_layer(graph, node_features, &params, attention_cache);

            // Calculate loss and feature gradients
            float loss = calculate_loss_and_gradients(node_features, training_data, feature_gradients);

            // Backward pass
            backward_pass(graph, node_features, &params, &gradients, attention_cache, feature_gradients);

            // Clip gradients
            clip_gradients(&gradients);

            // Update parameters with adaptive learning rate
            update_parameters(&params, &gradients, adaptive_lr);

            // Decay learning rate
            adaptive_lr *= 0.9f;

            // Clean up attention cache
            for (int i = 0; i < NUM_NODES; i++)
            {
                if (attention_cache[i] != NULL)
                {
                    free(attention_cache[i][0]);
                    free(attention_cache[i][1]);
                    free(attention_cache[i]);
                    attention_cache[i] = NULL;
                }
            }

            // Print progress
            printf("Epoch %d/%d: Loss = %.6f, LR = %.6f\n", epoch + 1, NUM_EPOCHS, loss, adaptive_lr);
        }

        // Calculate final time
        double end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;

        // Evaluate model on test set
        float auc = 0.0f;
        int correct_predictions = 0;
        int total_predictions = 0;

        // Simple evaluation on training set
        for (int p = 0; p < training_data->num_pairs; p++)
        {
            int node1 = training_data->node_pairs[p * 2];
            int node2 = training_data->node_pairs[p * 2 + 1];
            float target = training_data->labels[p];

            float prediction = predict_link(node1, node2, node_features);
            if ((prediction >= 0.5f && target == 1.0f) || (prediction < 0.5f && target == 0.0f))
            {
                correct_predictions++;
            }
            total_predictions++;
        }

        auc = (float)correct_predictions / total_predictions;

        // Write results to file
        fprintf(fp, "%d,%.6f,%.6f,%.6f\n", num_threads, elapsed_time,
                calculate_loss_and_gradients(node_features, training_data, feature_gradients), auc);

        // Print final results
        // printf("\nResults with %d threads:\n", num_threads);
        printf("  Time: %.6f seconds\n", elapsed_time);
        printf("  Accuracy: %.2f%% (%d/%d correct)\n",
               100.0f * correct_predictions / total_predictions,
               correct_predictions, total_predictions);

        // Test link prediction
        printf("\nLink prediction examples:\n");
        for (int i = 0; i < 5; i++)
        {
            int node1 = rand() % NUM_NODES;
            int node2 = rand() % NUM_NODES;
            float prob = predict_link(node1, node2, node_features);
            printf("  Node %d -> Node %d: %.2f%% probability\n", node1, node2, prob * 100.0f);
        }

        // Save model parameters
        save_parameters(&params);

        // Cleanup for this iteration
        for (int i = 0; i < NUM_NODES; i++)
        {
            free(graph[i].neighbors);
            if (attention_cache[i] != NULL)
            {
                free(attention_cache[i][0]);
                free(attention_cache[i][1]);
                free(attention_cache[i]);
            }
        }
        free(graph);
        free(attention_cache);
        free_matrix(node_features, NUM_NODES);
        free_matrix(feature_gradients, NUM_NODES);

        // Free training data
        free(training_data->node_pairs);
        free(training_data->labels);
        free(training_data);

        // Free model parameters
        free_matrix(params.attention_weights, NUM_FEATURES);
        free_matrix(params.weights, NUM_FEATURES);
        free(params.attention_bias);
        free(params.node_bias);

        // Free gradients
        free_matrix(gradients.attention_weights, NUM_FEATURES);
        free_matrix(gradients.weights, NUM_FEATURES);
        free(gradients.attention_bias);
        free(gradients.node_bias);
    }

    fclose(fp);
    printf("\nBenchmark results saved to benchmark_results.txt\n");

    return 0;
}