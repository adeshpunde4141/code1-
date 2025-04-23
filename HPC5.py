# Import required libraries
import tensorflow as tf
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the neural network architecture
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), 
        activation='relu', 
        input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Load and preprocess the MNIST dataset
def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize pixel values to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

# Training function with distributed data processing
def train_model(model, x_train, y_train, rank, size):
    # Split data across nodes
    n = len(x_train)
    chunk_size = n // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size
    
    # Handle remainder for last node
    if rank == size - 1:
        end = n
    
    x_train_chunk = x_train[start:end]
    y_train_chunk = y_train[start:end]
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model on local data
    model.fit(x_train_chunk, y_train_chunk, epochs=1, batch_size=32)
    
    # Evaluate local accuracy
    _, train_acc = model.evaluate(x_train_chunk, y_train_chunk, verbose=2)
    
    # Aggregate accuracy across all nodes
    global_train_acc = comm.allreduce(train_acc, op=MPI.SUM)
    return global_train_acc / size

# Main execution block
if __name__ == "__main__":
    # Create model and load data
    model = create_model()
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Training parameters
    epochs = 5
    
    # Training loop
    for epoch in range(epochs):
        # Distributed training
        train_acc = train_model(model, x_train, y_train, rank, size)
        
        # Evaluate on test data
        _, test_acc = model.evaluate(x_test, y_test, verbose=2)
        global_test_acc = comm.allreduce(test_acc, op=MPI.SUM)
        
        # Print results from rank 0 only
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy:  {global_test_acc/size:.4f}")
            print("-" * 40)