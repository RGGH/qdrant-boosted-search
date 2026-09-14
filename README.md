# H&M Personalized Fashion Recommendations - Enhanced Dataset

## Dataset Description

This dataset is a processed and enhanced version of the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) Kaggle competition dataset. The original dataset has been cleaned and augmented with pre-computed embeddings and accessible image URLs to facilitate fashion recommendation research and multimodal retrieval applications.

```
mkdir -p data && wget "https://huggingface.co/datasets/Qdrant/hm_ecommerce_products/resolve/main/hm_ecommerce_products_enriched.parquet" -O data/hm_products.parquet

uv add pandas
uv add fastparquet
```

### Dataset Summary

The H&M dataset contains rich product metadata, customer information, and transaction history from H&M's e-commerce platform. This enhanced version includes:
- **~106,000 fashion products** with detailed metadata
- **~1.37 million customers** with demographic information
- **Historical transaction data** for recommendation modeling
- **Product images** accessible via S3 URLs
- **Pre-computed dense embeddings** (BGE-small-en-1.5)
- **Pre-computed sparse embeddings** (SPLADEv1)

### Languages

- English (product descriptions and metadata)

## Enhancements Over Original Dataset

This processed version includes the following improvements:

1. **Data Cleaning**: Removed all null values for improved data quality
2. **Image Accessibility**: All product images uploaded to S3 with direct URL access via `image_url` field
3. **Dense Embeddings**: Pre-computed [BGE-small-en-1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings for efficient semantic search
4. **Sparse Embeddings**: Pre-computed [SPLADEv1](https://github.com/naver/splade) embeddings for keyword-based retrieval

## Dataset Structure

### Data Instances

Each article (product) record contains:

```json
{
  "article_id": "108775015",
  "product_code": "108775",
  "prod_name": "Strap top",
  "product_type_no": 253,
  "product_type_name": "Vest top",
  "product_group_name": "Garment Upper body",
  "graphical_appearance_no": 1010016,
  "graphical_appearance_name": "Solid",
  "colour_group_code": 9,
  "colour_group_name": "Black",
  "perceived_colour_value_id": 4,
  "perceived_colour_value_name": "Dark",
  "perceived_colour_master_id": 5,
  "perceived_colour_master_name": "Black",
  "department_no": 1676,
  "department_name": "Jersey Basic",
  "index_code": "A",
  "index_name": "Ladieswear",
  "index_group_no": 1,
  "index_group_name": "Ladieswear",
  "section_no": 16,
  "section_name": "Womens Everyday Basics",
  "garment_group_no": 1002,
  "garment_group_name": "Jersey Basic",
  "detail_desc": "Fitted strap top in soft jersey with narrow shoulder straps.",
  "image_url": "https://your-bucket.s3.amazonaws.com/108775015.jpg",
  "bge_embedding": [0.123, -0.456, ...],  // 384-dimensional vector
  "splade_embedding": {"token_id": weight, ...}  // Sparse vector representation
}
```

### Data Fields

#### Article Metadata
- `article_id` (string): Unique product identifier
- `product_code` (string): Product family code
- `prod_name` (string): Product name
- `product_type_no` (int): Product type numeric code
- `product_type_name` (string): Product type description
- `product_group_name` (string): High-level product category
- `graphical_appearance_no` (int): Pattern code
- `graphical_appearance_name` (string): Pattern description (e.g., "Solid", "Stripe")
- `colour_group_code` (int): Color category code
- `colour_group_name` (string): Color category name
- `perceived_colour_value_id` (int): Color brightness/darkness code
- `perceived_colour_value_name` (string): Color brightness description
- `perceived_colour_master_id` (int): Master color category code
- `perceived_colour_master_name` (string): Master color name
- `department_no` (int): Department code
- `department_name` (string): Department name
- `index_code` (string): Index category code
- `index_name` (string): Index category (e.g., "Ladieswear", "Menswear")
- `index_group_no` (int): Index group numeric code
- `index_group_name` (string): Index group name
- `section_no` (int): Section code
- `section_name` (string): Section description
- `garment_group_no` (int): Garment group code
- `garment_group_name` (string): Garment group name
- `detail_desc` (string): Detailed product description

#### Enhanced Fields
- `image_url` (string): Direct S3 URL to product image
- `bge_embedding` (list[float]): 384-dimensional dense embedding from BGE-small-en-1.5
- `splade_embedding` (dict): Sparse embedding from SPLADEv1 with token weights

### Data Splits

The dataset maintains the original temporal structure:
- Training period: 2018-09-20 to 2020-09-22
- Prediction target: Week of 2020-09-23 to 2020-09-29

## Dataset Creation

### Source Data

The original dataset was created by H&M Group for a Kaggle competition aimed at developing personalized fashion recommendation systems. The data represents real customer transactions and product catalog information.

### Processing Pipeline

1. **Data Loading**: Original CSV files from Kaggle competition
2. **Null Removal**: Filtered out records with missing critical fields
3. **Image Upload**: Product images uploaded to S3 for reliable access
4. **Embedding Generation**:
   - **BGE-small-en-1.5**: Dense embeddings generated from product descriptions and metadata
   - **SPLADEv1**: Sparse embeddings for efficient lexical matching
5. **Quality Validation**: Verified embedding dimensions and data integrity

### Embedding Models

#### BGE-small-en-1.5
- **Model**: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Dimension**: 384
- **Use Case**: Dense semantic retrieval, similarity search
- **Input**: Product descriptions and relevant metadata fields

#### SPLADEv1
- **Model**: SPLADE (Sparse Lexical and Expansion Model)
- **Type**: Learned sparse representation
- **Use Case**: Keyword-based retrieval, interpretable search

## Uses

### Direct Use

This dataset is ideal for:
- **Fashion recommendation systems**: Product-to-product and user-to-product recommendations
- **Multimodal retrieval**: Combining text and image features for search
- **Embedding evaluation**: Benchmarking dense and sparse retrieval methods
- **Fashion trend analysis**: Understanding product categorization and customer preferences
- **Vector database applications**: Testing and demonstrating retrieval systems (e.g., with Qdrant, Pinecone, Weaviate)

### Use Cases

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/hm-fashion-enhanced")

# Example 1: Semantic search using BGE embeddings
query_embedding = model.encode("casual summer dress")
# Use embeddings for similarity search in your vector database

# Example 2: Hybrid search combining dense and sparse
# Combine BGE dense embeddings with SPLADE sparse for best results

# Example 3: Fashion recommendation
# Use transaction history + embeddings for personalized recommendations
```

## Considerations for Using the Data

### Social Impact

Fashion recommendation systems can influence purchasing decisions and consumer behavior. Consider:
- **Diversity**: Ensure recommendations don't reinforce narrow beauty standards
- **Inclusivity**: Product recommendations should serve diverse customer demographics
- **Sustainability**: Consider environmental impact of fast fashion recommendations

### Limitations

- **Temporal Scope**: Data covers 2018-2020; fashion trends evolve rapidly
- **Geographic Bias**: Primarily represents European H&M markets
- **Demographic Representation**: Customer demographics may not represent global population
- **Cold Start**: Limited data for new products and customers
- **Null Removal**: Some products excluded due to missing data

### Privacy

- Customer data has been anonymized
- No personally identifiable information (PII) included
- Transaction history anonymized with customer IDs

## Additional Information

### Dataset Curators

Original dataset: H&M Group via Kaggle
Enhanced version: Qdrant

### Licensing Information

The dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license, following the original Kaggle competition terms.

### Citation

If you use this dataset, please cite both the original competition and this enhanced version:

```bibtex
@misc{hm-fashion-kaggle,
  title={H\&M Personalized Fashion Recommendations},
  author={H\&M Group},
  year={2022},
  howpublished={Kaggle Competition},
  url={https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations}
}

@misc{hm-fashion-enhanced,
  title={H\&M Personalized Fashion Recommendations - Enhanced with Embeddings},
  author={[Your Name]},
  year={2024},
  howpublished={HuggingFace Datasets},
  url={[your dataset URL]}
}
```

### Contributions

Contributions and feedback are welcome! Please open an issue or pull request if you find any problems or have suggestions for improvements.

### Acknowledgments

- H&M Group for providing the original dataset
- Kaggle for hosting the competition
- BAAI for the BGE embedding model
- NAVER for the SPLADE sparse embedding model


