import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
import nltk
import os


# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass


download_nltk_data()


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model classes from your notebook
class SentenceEncoder(nn.Module):
    def __init__(self, model_name='law-ai/InLegalBERT', hidden_dim=768):
        super(SentenceEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        
        # Freeze encoder
        for param in self.model.parameters():
            param.requires_grad = False
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, sentences):
        encoded = self.tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            model_output = self.model(**encoded)
            embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
        
        return embeddings


class MultiAspectPolicyNetwork(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_aspects=5, dropout=0.5):
        super(MultiAspectPolicyNetwork, self).__init__()
        self.num_aspects = num_aspects
        self.aspects = ['facts', 'analysis', 'argument', 'judgement', 'statute']
        self.hidden_dim = hidden_dim
        
        # Shared LSTM encoder
        self.shared_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                                   bidirectional=True, batch_first=True, dropout=dropout)
        
        # Position and aspect embeddings
        self.position_embedding = nn.Embedding(500, 64)
        self.aspect_embedding = nn.Embedding(num_aspects, hidden_dim * 2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=4, 
                                              dropout=dropout, batch_first=True)
        
        # Aspect-specific heads
        self.aspect_heads = nn.ModuleDict()
        for aspect in self.aspects:
            self.aspect_heads[aspect] = nn.Sequential(
                nn.Linear((hidden_dim * 2) + 64 + (hidden_dim * 2), 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1)
            )
    
    def forward(self, sentence_embeddings, positions, aspect_idx):
        # LSTM encoding
        lstm_out, _ = self.shared_lstm(sentence_embeddings)
        
        # Aspect embedding query
        aspect_emb_query = self.aspect_embedding(torch.tensor([aspect_idx], device=device))
        aspect_emb_query = aspect_emb_query.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        
        # Apply attention
        attended_out, _ = self.attention(aspect_emb_query, lstm_out, lstm_out)
        
        # Combine features
        combined_lstm = lstm_out + attended_out
        pos_emb = self.position_embedding(positions)
        aspect_emb_concat = self.aspect_embedding(torch.tensor([aspect_idx], device=device))
        aspect_emb_concat = aspect_emb_concat.unsqueeze(1).expand(-1, sentence_embeddings.size(1), -1)
        
        combined = torch.cat([combined_lstm, pos_emb, aspect_emb_concat], dim=-1)
        
        # Aspect-specific scoring
        aspect_name = self.aspects[aspect_idx]
        logits = self.aspect_heads[aspect_name](combined).squeeze(-1)
        
        return logits


class UnsupervisedRLAgent:
    def __init__(self, encoder, policy):
        self.encoder = encoder.to(device)
        self.policy = policy.to(device)
        self.aspects = ['facts', 'analysis', 'argument', 'judgement', 'statute']
        self.aspect_summary_ratios = {
            'facts': 0.12,
            'analysis': 0.12,
            'argument': 0.08,
            'judgement': 0.06,
            'statute': 0.08
        }
        self.min_summary_sentences = 3
    
    def preprocess_document(self, judgment_text):
        sentences = sent_tokenize(judgment_text)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 5]
        return sentences
    
    def encode_sentences(self, sentences):
        if len(sentences) == 0:
            return torch.zeros(1, self.encoder.hidden_dim).to(device)
        
        batch_size = 32
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            emb = self.encoder(batch)
            embeddings.append(emb)
        
        return torch.cat(embeddings, dim=0)
    
    def generate_summaries(self, judgment, temperature=0.3):
        self.policy.eval()
        
        with torch.no_grad():
            sentences = self.preprocess_document(judgment)
            
            if len(sentences) < 3:
                return {aspect: ". ".join(sentences) for aspect in self.aspects}
            
            if len(sentences) > 490:
                sentences = sentences[:490]
            
            sentence_embeddings = self.encode_sentences(sentences)
            sentence_embeddings = sentence_embeddings.unsqueeze(0)
            positions = torch.arange(len(sentences), device=device).unsqueeze(0)
            
            summaries = {}
            for aspect_idx, aspect in enumerate(self.aspects):
                logits = self.policy(sentence_embeddings, positions, aspect_idx).squeeze(0)
                
                aspect_ratio = self.aspect_summary_ratios[aspect]
                num_select = max(self.min_summary_sentences, int(len(sentences) * aspect_ratio))
                
                topk_indices = torch.topk(logits, k=num_select).indices
                topk_indices = sorted(topk_indices.cpu().numpy())
                
                summary = ". ".join([sentences[i] for i in topk_indices])
                summaries[aspect] = summary
            
            return summaries


# Load model function - automatically loads on startup
@st.cache_resource
def load_model(checkpoint_path="inlegalbert-50-unsupervised_legal_summarization.pt"):
    """Load the trained RL model automatically on app startup"""
    if not os.path.exists(checkpoint_path):
        return None, None, f"Model file not found: {checkpoint_path}"
    
    try:
        encoder = SentenceEncoder(model_name='law-ai/InLegalBERT', hidden_dim=768)
        policy = MultiAspectPolicyNetwork(input_dim=768, hidden_dim=256, num_aspects=5, dropout=0.5)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        # Initialize agent
        agent = UnsupervisedRLAgent(encoder=encoder, policy=policy)
        agent.policy.eval()
        
        return agent, checkpoint, None
    except Exception as e:
        return None, None, str(e)


# Streamlit App
def main():
    st.set_page_config(
        page_title="Legal Document Summarizer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal Document Summarizer")
    st.markdown("**Reinforcement Learning-based Multi-Aspect Legal Judgment Summarization**")
    
    # Automatically load model on startup
    model_path = "inlegalbert-50-unsupervised_legal_summarization.pt"
    agent, checkpoint, error = load_model(model_path)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Model Status")
        
        if error:
            st.error(f"❌ {error}")
            st.warning("⚠️ Please ensure the model file is in the correct location:")
            st.code(model_path)
        elif agent is not None:
            st.success("✅ Model loaded successfully!")
            st.info(f"**Model:** {model_path}")
            st.info(f"**Val Reward:** {checkpoint['val_reward']:.4f}")
            st.info(f"**Device:** {device}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a **Reinforcement Learning** approach to generate aspect-wise summaries:
        - **Facts**: Key factual information
        - **Analysis**: Legal analysis
        - **Argument**: Arguments presented
        - **Judgement**: Court's decision
        - **Statute**: Relevant statutes
        """)
    
    # Main content
    if agent is None:
        st.warning("⚠️ Model not loaded. Please check the model file path and ensure it exists.")
        st.info(f"Expected model path: `{model_path}`")
        return
    
    # Text input options
    input_method = st.radio("Choose input method:", ["📝 Paste Text", "📄 Upload File"])
    
    judgment_text = ""
    
    if input_method == "📝 Paste Text":
        judgment_text = st.text_area(
            "Enter legal judgment text:",
            height=300,
            placeholder="Paste the legal judgment text here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file is not None:
            judgment_text = uploaded_file.read().decode('utf-8')
            st.text_area("File content preview:", judgment_text[:1000] + "...", height=200, disabled=True)
    
    # Generate summaries button
    if st.button("🎯 Generate Summaries", type="primary"):
        if not judgment_text.strip():
            st.error("❌ Please enter or upload a legal judgment text.")
            return
        
        with st.spinner("🔄 Generating aspect-wise summaries..."):
            try:
                summaries = agent.generate_summaries(judgment_text)
                
                # Display results
                st.success("✅ Summaries generated successfully!")
                
                # Document statistics
                sentences = agent.preprocess_document(judgment_text)
                st.markdown("### 📊 Document Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Sentences", len(sentences))
                col2.metric("Total Words", sum([len(s.split()) for s in sentences]))
                
                total_summary_words = sum([len(summaries[a].split()) for a in agent.aspects])
                col3.metric("Summary Words", total_summary_words)
                
                # Aspect-wise summaries
                st.markdown("### 📝 Aspect-wise Summaries")
                
                aspect_icons = {
                    'facts': '📌',
                    'analysis': '🔍',
                    'argument': '💬',
                    'judgement': '⚖️',
                    'statute': '📜'
                }
                
                for aspect in agent.aspects:
                    with st.expander(f"{aspect_icons[aspect]} **{aspect.upper()}** ({len(summaries[aspect].split())} words)", expanded=True):
                        st.markdown(summaries[aspect])
                        
                        # Compression ratio
                        original_words = sum([len(s.split()) for s in sentences])
                        aspect_words = len(summaries[aspect].split())
                        compression = (aspect_words / original_words) * 100
                        st.caption(f"📉 Compression: {compression:.1f}%")
                
                # Overall statistics
                st.markdown("### 📈 Overall Statistics")
                original_words = sum([len(s.split()) for s in sentences])
                overall_compression = (total_summary_words / original_words) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Overall Compression Ratio", f"{overall_compression:.2f}%")
                col2.metric("Avg Words per Aspect", f"{total_summary_words / len(agent.aspects):.0f}")
                
            except Exception as e:
                st.error(f"❌ Error generating summaries: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
