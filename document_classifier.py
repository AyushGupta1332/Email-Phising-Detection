"""
Document Classifier Module using RoBERTa + LoRA
Classifies text content from documents (PDF, DOCX, CSV, Excel, TXT) as sensitive or non-sensitive.
Uses majority voting for long documents.
"""

import os
import io
import torch
import pandas as pd
import chardet
import PyPDF2
from docx import Document
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.nn.functional import softmax
from nltk.tokenize import sent_tokenize
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
_model = None
_tokenizer = None
_device = None

def initialize_document_classifier():
    """
    Initialize the RoBERTa + LoRA document classifier model.
    Call this once at application startup.
    """
    global _model, _tokenizer, _device
    
    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Document classifier using device: {_device}")
        
        # Load tokenizer
        _tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        # Load base model with LoRA configuration
        base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none"
        )
        _model = get_peft_model(base_model, lora_config)
        
        # Load trained weights
        model_path = os.path.join(
            os.path.dirname(__file__),
            "Data Classification File and Model",
            "best_roberta_model_2.2M_1_Epoc.pt"
        )
        _model.load_state_dict(torch.load(model_path, map_location=_device))
        _model.to(_device)
        _model.eval()
        
        logger.info("Document classifier (RoBERTa + LoRA) initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize document classifier: {e}")
        return False


# ----------- TEXT EXTRACTION FUNCTIONS -----------

def safe_read_csv_data(file_data):
    """Read CSV from bytes data with encoding detection."""
    try:
        encoding = chardet.detect(file_data[:10000])['encoding'] or 'utf-8'
        return pd.read_csv(io.BytesIO(file_data), encoding=encoding)
    except:
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                return pd.read_csv(io.BytesIO(file_data), encoding=encoding)
            except:
                continue
        return pd.read_csv(io.BytesIO(file_data), encoding='utf-8', errors='ignore')


def analyze_tabular_data(df):
    """Convert tabular data to descriptive sentences for classification."""
    if df.empty:
        return ["Empty dataset with no data"]
    
    sentences = []
    df = df.head(1000) if len(df) > 1000 else df
    df = df.iloc[:, :50] if len(df.columns) > 50 else df
    
    # Create descriptive sentences from data
    for idx, row in df.head(10).iterrows():
        row_data = []
        for col, val in row.items():
            if pd.notna(val):
                val_str = str(val)[:100]
                row_data.append(f"{col}: {val_str}")
        if row_data:
            sentence = "Record contains " + ", ".join(row_data[:5])
            sentences.append(sentence)
    
    sentences.append(f"Dataset has {len(df)} rows and {len(df.columns)} columns")
    
    # Check for sensitive column names
    sensitive_keywords = ['name', 'email', 'phone', 'address', 'ssn', 'id', 'password', 'salary', 'credit', 'account']
    sensitive_cols = [col for col in df.columns if any(keyword in str(col).lower() for keyword in sensitive_keywords)]
    
    if sensitive_cols:
        sentences.append(f"Dataset contains potentially sensitive columns: {', '.join(sensitive_cols[:10])}")
    
    return sentences[:20]


def extract_text_from_document_data(file_data, filename):
    """
    Extract text from document bytes based on file extension.
    Supports: .txt, .pdf, .docx, .csv, .xlsx, .xls
    """
    extension = Path(filename).suffix.lower()
    text = ""
    
    try:
        if extension == '.txt':
            # Try multiple encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    text = file_data.decode(encoding, errors='ignore')[:50000]
                    break
                except:
                    continue
                    
        elif extension == '.docx':
            try:
                doc = Document(io.BytesIO(file_data))
                paragraphs = [para.text for para in doc.paragraphs if para.text.strip()][:100]
                text = "\n\n".join(paragraphs)
            except Exception as e:
                logger.warning(f"Error reading DOCX: {e}")
                text = "Error reading DOCX file"
            
        elif extension == '.pdf':
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data))
                pages_to_read = min(len(pdf_reader.pages), 20)
                for i in range(pages_to_read):
                    try:
                        page_text = pdf_reader.pages[i].extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except:
                        continue
            except Exception as e:
                logger.warning(f"Error reading PDF: {e}")
                text = "Error reading PDF file"
                
        elif extension == '.csv':
            try:
                df = safe_read_csv_data(file_data)
                if df is not None:
                    df = df.head(1000)
                    sentences = analyze_tabular_data(df)
                    text = ". ".join(sentences) + "."
                else:
                    text = "Error reading CSV file"
            except Exception as e:
                logger.warning(f"Error reading CSV: {e}")
                text = "Error processing CSV file"
            
        elif extension in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(io.BytesIO(file_data), nrows=1000)
                sentences = analyze_tabular_data(df)
                text = ". ".join(sentences) + "."
            except Exception as e:
                logger.warning(f"Error reading Excel: {e}")
                text = "Error reading Excel file: contains tabular data"
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        text = ""
    
    return text[:20000] if text else ""


# ----------- CLASSIFICATION FUNCTIONS -----------

def safe_tokenize(text, max_length=512):
    """Safely tokenize text with error handling."""
    global _tokenizer
    try:
        text = str(text)[:10000]
        return _tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    except:
        return {
            'input_ids': torch.zeros((1, max_length), dtype=torch.long),
            'attention_mask': torch.zeros((1, max_length), dtype=torch.long)
        }


def classify_single_text(text):
    """Classify a single piece of text. Returns (prediction, confidence)."""
    global _model, _device
    try:
        if not text or not text.strip():
            return 0, 0.5
        
        inputs = safe_tokenize(text, max_length=512)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs.max().item()
            
        return pred, conf
    except Exception as e:
        logger.warning(f"Error in classify_single_text: {e}")
        return 0, 0.5


def classify_with_majority_voting(text, token_threshold=500, majority_threshold=0.8):
    """
    Classify text using majority voting for long documents.
    For short texts, classifies directly. For long texts, splits into sentences
    and uses majority voting with configurable threshold.
    
    Returns: (prediction, confidence)
        - prediction: 0 = Non-Sensitive, 1 = Sensitive
        - confidence: Average confidence score
    """
    global _tokenizer
    try:
        if not text or not text.strip():
            return 0, 0.5
        
        text = text[:20000]
        
        # Get token count
        try:
            token_count = len(_tokenizer.encode(text[:10000], add_special_tokens=False))
        except:
            token_count = len(text.split()) * 1.3
        
        # Single classification for short texts
        if token_count <= token_threshold:
            pred, conf = classify_single_text(text)
            return pred, conf
        
        # Majority voting for long texts
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return classify_single_text(text[:2000])
        
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 10]
        if not meaningful_sentences:
            meaningful_sentences = sentences
        
        max_sentences = 50
        if len(meaningful_sentences) > max_sentences:
            meaningful_sentences = meaningful_sentences[:max_sentences]
        
        sensitive_count = 0
        total_confidence = 0
        
        for sentence in meaningful_sentences:
            pred, conf = classify_single_text(sentence)
            if pred == 1:
                sensitive_count += 1
            total_confidence += conf
        
        total_sentences = len(meaningful_sentences)
        
        # Use manual threshold instead of fixed 50%
        required_sensitive_count = total_sentences * majority_threshold
        
        if sensitive_count >= required_sensitive_count:
            final_prediction = 1
        else:
            final_prediction = 0
        
        avg_confidence = total_confidence / total_sentences if total_sentences > 0 else 0.5
        
        return final_prediction, avg_confidence
        
    except Exception as e:
        logger.error(f"Error in classify_with_majority_voting: {e}")
        return 0, 0.5


def classify_document(file_data, filename):
    """
    Main function to classify a document as sensitive or non-sensitive.
    
    Args:
        file_data: Bytes of the file content
        filename: Name of the file (to determine extension)
    
    Returns:
        str: 'sensitive' or 'non-sensitive'
    """
    global _model
    
    # Check if model is initialized
    if _model is None:
        logger.warning("Document classifier not initialized, initializing now...")
        if not initialize_document_classifier():
            logger.error("Failed to initialize document classifier, returning non-sensitive")
            return 'non-sensitive'
    
    try:
        # Extract text from document
        text = extract_text_from_document_data(file_data, filename)
        
        if not text or text.strip() == '':
            return 'non-sensitive'
        
        # Classify using majority voting
        prediction, confidence = classify_with_majority_voting(text, token_threshold=500)
        
        logger.debug(f"Document {filename}: prediction={prediction}, confidence={confidence:.4f}")
        
        # Convert to string label
        return 'sensitive' if prediction == 1 else 'non-sensitive'
        
    except Exception as e:
        logger.error(f"Error classifying document {filename}: {e}")
        return 'non-sensitive'


def classify_text_content(text):
    """
    Classify plain text content as sensitive or non-sensitive.
    Use this when you already have extracted text.
    
    Args:
        text: Plain text string
    
    Returns:
        str: 'sensitive' or 'non-sensitive'
    """
    global _model
    
    # Check if model is initialized
    if _model is None:
        logger.warning("Document classifier not initialized, initializing now...")
        if not initialize_document_classifier():
            logger.error("Failed to initialize document classifier, returning non-sensitive")
            return 'non-sensitive'
    
    try:
        if not text or text.strip() == '':
            return 'non-sensitive'
        
        prediction, confidence = classify_with_majority_voting(text, token_threshold=500)
        
        return 'sensitive' if prediction == 1 else 'non-sensitive'
        
    except Exception as e:
        logger.error(f"Error classifying text content: {e}")
        return 'non-sensitive'
