
# Configuration Instructions for Benchmarking LLMS

This guide provides detailed instructions on setting up and running experiments in the `benchmark_llms` repository.

## Getting Started

### Cloning and Initial Setup

1. Clone this repository:
   ```
   git clone [repository URL]
   ```
2. Navigate to the repository directory:
   ```
   cd benchmarking_llms
   ```
3. Run the setup script with sudo:
   ```
   ./configuration.sh
   ```

## Repository Structure

The repository consists of several subdirectories that serve specific purposes:

- **llms** and **vec_db**: Default locations for OLLaMa and Qdrant storage.
- **knowledge_base**: Store your knowledge bases for RAG here. Ensure you create a new subdirectory for each knowledge base.
- **logs**: Stores the results of experiments. Logs for metric data are saved as `evaluation_with_[rag|context].log`, and logs for LLMS responses are saved as `response_with_[rag|context].log`.
- **queries**: Stores query inputs for LLMS. Filenames should follow the format `(prefix)_query.txt` and `(prefix)_context.txt`. Ensure there is a one-to-one correspondence between the lines in `query.txt`, `context.txt`, and `response.log`. Note that `context` and `rag` are mutually exclusive; if `rag` is applied, the `context` file will be ignored.
- **templates**: Template files located here must have two attributes: `context` and `input`.
  - `context`: External knowledge used in the prompt, sourced either from RAG or `context.txt`.
  - `input`: The query posed to the LLMS, sourced from `query.txt`.

## Configuring Experiments

Modify the `experiment_config.yml` file to set up your experiments, including specifying LLMS options and variables.

## Important Notes

1. **Query Input Function**: Implement the `read_in_query` function in `evaluation.py`. If needed, you can create a new file and import it. The format of the return value is annotated in `evaluation.py`.
2. **Running Experiments**: Execute all experiments within the `benchmark_llms` directory for consistency.
3. **Demo**: The repo provides a demo data in queries and templates (thanks Qiming). You can run `evaluation.py` after finishing configuration.