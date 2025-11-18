import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
import nltk
import os
import gc


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


# Model classes
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
    def __init__(self, input_dim=768, hidden_dim=256, num_aspects=5, dropout=0.5, max_positions=500):
        super(MultiAspectPolicyNetwork, self).__init__()
        self.num_aspects = num_aspects
        self.aspects = ['facts', 'analysis', 'argument', 'judgement', 'statute']
        self.hidden_dim = hidden_dim
        self.max_positions = max_positions
        
        # Shared LSTM encoder
        self.shared_lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                                   bidirectional=True, batch_first=True, dropout=dropout)
        
        # Position and aspect embeddings
        self.position_embedding = nn.Embedding(max_positions, 64)
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
        lstm_out, _ = self.shared_lstm(sentence_embeddings)
        aspect_emb_query = self.aspect_embedding(torch.tensor([aspect_idx], device=device))
        aspect_emb_query = aspect_emb_query.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        attended_out, _ = self.attention(aspect_emb_query, lstm_out, lstm_out)
        combined_lstm = lstm_out + attended_out
        pos_emb = self.position_embedding(positions)
        aspect_emb_concat = self.aspect_embedding(torch.tensor([aspect_idx], device=device))
        aspect_emb_concat = aspect_emb_concat.unsqueeze(1).expand(-1, sentence_embeddings.size(1), -1)
        combined = torch.cat([combined_lstm, pos_emb, aspect_emb_concat], dim=-1)
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
        self.max_positions = policy.max_positions
    
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
            
            max_sent = self.max_positions - 10
            if len(sentences) > max_sent:
                sentences = sentences[:max_sent]
            
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


def detect_model_architecture(checkpoint_path):
    """Detect the position embedding size from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['policy_state_dict']
        pos_emb_weight = state_dict['position_embedding.weight']
        max_positions = pos_emb_weight.shape[0]
        return max_positions
    except Exception as e:
        print(f"Error detecting architecture: {e}")
        return 500


# Load both models with automatic architecture detection
@st.cache_resource
def load_both_models(model1_path, model2_path):
    """Load both models with different architectures"""
    models = {}
    checkpoints = {}
    errors = {}
    
    for model_name, path in [("Model 1", model1_path), ("Model 2", model2_path)]:
        if not os.path.exists(path):
            models[model_name] = None
            checkpoints[model_name] = None
            errors[model_name] = f"File not found: {path}"
            continue
        
        try:
            max_positions = detect_model_architecture(path)
            encoder = SentenceEncoder(model_name='law-ai/InLegalBERT', hidden_dim=768)
            policy = MultiAspectPolicyNetwork(
                input_dim=768, 
                hidden_dim=256, 
                num_aspects=5, 
                dropout=0.5,
                max_positions=max_positions
            )
            
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            
            agent = UnsupervisedRLAgent(encoder=encoder, policy=policy)
            agent.policy.eval()
            
            models[model_name] = agent
            checkpoints[model_name] = checkpoint
            errors[model_name] = None
            
        except Exception as e:
            models[model_name] = None
            checkpoints[model_name] = None
            errors[model_name] = str(e)
    
    return models, checkpoints, errors


def display_gpu_stats():
    """Display GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        st.sidebar.write("### 🎮 GPU Memory")
        progress_value = min(allocated / total, 1.0)
        st.sidebar.progress(progress_value)
        st.sidebar.write(f"**Allocated:** {allocated:.2f} GB")
        st.sidebar.write(f"**Reserved:** {reserved:.2f} GB")
        st.sidebar.write(f"**Total:** {total:.2f} GB")
        st.sidebar.write(f"**Free:** {total - reserved:.2f} GB")
        
        if allocated / total > 0.85:
            st.sidebar.warning("⚠️ GPU memory usage is high!")
    else:
        st.sidebar.info("Running on CPU")


def main():
    st.set_page_config(
        page_title="Dual Model Legal Summarizer",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Dual Model Legal Document Summarizer")
    st.markdown("**Reinforcement Learning-based Multi-Aspect Legal Judgment Summarization**")
    st.markdown("*Running 2 models simultaneously for comparison*")
    
    # UPDATE THESE PATHS
    model1_path = "inlegalbert-50-unsupervised_legal_summarization.pt"
    model2_path = "model(400data).pt" 
    
    models, checkpoints, errors = load_both_models(model1_path, model2_path)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Model Status")
        st.info(f"**Device:** {device}")
        if torch.cuda.is_available():
            st.success(f"**GPU:** {torch.cuda.get_device_name(0)}")
        
        st.markdown("---")
        
        # Model 1 status
        st.subheader("Model 1")
        if errors["Model 1"]:
            st.error(f"❌ {errors['Model 1']}")
        elif models["Model 1"] is not None:
            st.success("✅ Loaded")
            st.info(f"Val Reward: {checkpoints['Model 1']['val_reward']:.4f}")
        
        st.markdown("---")
        
        # Model 2 status
        st.subheader("Model 2")
        if errors["Model 2"]:
            st.error(f"❌ {errors['Model 2']}")
        elif models["Model 2"] is not None:
            st.success("✅ Loaded")
            st.info(f"Val Reward: {checkpoints['Model 2']['val_reward']:.4f}")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        **Aspects:**
        - 📌 Facts
        - 🔍 Analysis
        - 💬 Argument
        - ⚖️ Judgement
        - 📜 Statute
        """)

        st.markdown("---")

        display_gpu_stats()
        
        if st.button("🧹 Clear GPU Cache"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                st.success("Cache cleared!")
                st.rerun()
        
        
        
    
    if models["Model 1"] is None and models["Model 2"] is None:
        st.error("⚠️ No models loaded. Please check the model file paths.")
        st.info(f"Expected Model 1: `{model1_path}`")
        st.info(f"Expected Model 2: `{model2_path}`")
        return
    
    available_models = [name for name, model in models.items() if model is not None]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_model = st.selectbox("🎯 Select Model", available_models)
    with col2:
        compare_mode = st.checkbox("Compare Both", help="Generate from both models")
    
    input_method = st.radio("Choose input method:", ["📝 Paste Text", "📄 Upload File"])
    
    judgment_text = ""
    if input_method == "📝 Paste Text":
        judgment_text = st.text_area("Enter legal judgment text:", height=250, 
                                     placeholder="Paste the legal judgment text here...")
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file is not None:
            judgment_text = uploaded_file.read().decode('utf-8')
            with st.expander("📄 File content preview"):
                st.text_area("Preview", judgment_text[:1000] + "...", height=200, disabled=True)
    
    if st.button("🎯 Generate Summaries", type="primary"):
        if not judgment_text.strip():
            st.error("❌ Please enter or upload a legal judgment text.")
            return
        
        # Compare mode: use both models
        if compare_mode and len(available_models) == 2:
            st.markdown("---")
            st.markdown("## 📊 Model Comparison")
            
            cols = st.columns(2)
            
            for idx, model_name in enumerate(available_models):
                with cols[idx]:
                    st.subheader(f"🤖 {model_name}")
                    
                    with st.spinner(f"Generating summaries with {model_name}..."):
                        try:
                            agent = models[model_name]
                            summaries = agent.generate_summaries(judgment_text)
                            
                            # Document statistics
                            sentences = agent.preprocess_document(judgment_text)
                            total_original_words = sum([len(s.split()) for s in sentences])
                            total_summary_words = sum([len(summaries[a].split()) for a in agent.aspects])
                            
                            # Display stats
                            st.markdown("### 📊 Statistics")
                            stat_col1, stat_col2, stat_col3 = st.columns(3)
                            stat_col1.metric("Sentences", len(sentences))
                            stat_col2.metric("Original(words)", f"{total_original_words}")
                            stat_col3.metric("Summary(words)", f"{total_summary_words}")
                            
                            compression_ratio = (total_summary_words / total_original_words) * 100
                            st.metric("Compression Ratio", f"{compression_ratio:.2f}%")
                            
                            st.markdown("---")
                            
                            # Aspect-wise summaries
                            aspect_icons = {
                                'facts': '📌',
                                'analysis': '🔍',
                                'argument': '💬',
                                'judgement': '⚖️',
                                'statute': '📜'
                            }
                            
                            for aspect in agent.aspects:
                                aspect_words = len(summaries[aspect].split())
                                with st.expander(f"{aspect_icons[aspect]} {aspect.upper()} ({aspect_words} words)", expanded=False):
                                    st.markdown(summaries[aspect])
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.exception(e)
                    
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Single model mode
        else:
            with st.spinner(f"🔄 Generating summaries with {selected_model}..."):
                try:
                    agent = models[selected_model]
                    summaries = agent.generate_summaries(judgment_text)
                    
                    st.success("✅ Summaries generated successfully!")
                    
                    # Document statistics
                    sentences = agent.preprocess_document(judgment_text)
                    total_original_words = sum([len(s.split()) for s in sentences])
                    total_summary_words = sum([len(summaries[a].split()) for a in agent.aspects])
                    
                    st.markdown("### 📊 Document Statistics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Sentences", len(sentences))
                    col2.metric("Original Words", total_original_words)
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
                        aspect_words = len(summaries[aspect].split())
                        aspect_compression = (aspect_words / total_original_words) * 100
                        
                        with st.expander(f"{aspect_icons[aspect]} **{aspect.upper()}** ({aspect_words} words)", expanded=True):
                            st.markdown(summaries[aspect])
                            st.caption(f"📉 Compression: {aspect_compression:.1f}%")
                    
                    # Overall statistics
                    st.markdown("### 📈 Overall Statistics")
                    overall_compression = (total_summary_words / total_original_words) * 100
                    
                    stat_col1, stat_col2 = st.columns(2)
                    stat_col1.metric("Overall Compression", f"{overall_compression:.2f}%")
                    stat_col2.metric("Avg Words/Aspect", f"{total_summary_words / len(agent.aspects):.0f}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating summaries: {str(e)}")
                    st.exception(e)
                
                torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
