import os
import chromadb
from chromadb.utils import embedding_functions

def build_vector_db():
    # 1. 获取知识库路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(base_dir, 'knowledge_base')
    db_dir = os.path.join(base_dir, 'chroma_db')
    
    if not os.path.exists(kb_dir):
        print(f"错误：找不到知识库目录 {kb_dir}")
        return

    # 2. 初始化 ChromaDB (持久化到本地)
    print("正在初始化 ChromaDB...")
    client = chromadb.PersistentClient(path=db_dir)
    
    # 使用轻量级的默认嵌入模型 (all-MiniLM-L6-v2)
    # 在生产环境中，可以替换为 OpenAI 或阿里云 DashScope 的嵌入模型以获得更好的中文效果
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 获取或创建 Collection
    collection_name = "marketing_scripts"
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
        
    collection = client.create_collection(
        name=collection_name,
        embedding_function=sentence_transformer_ef
    )
    
    # 3. 读取 Markdown 文件并切分 (Chunking)
    documents = []
    metadatas = []
    ids = []
    
    print("正在读取并切分话术文档...")
    doc_id_counter = 1
    
    for filename in os.listdir(kb_dir):
        if filename.endswith('.md') or filename.endswith('.txt'):
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 简单的按段落切分 (以双换行符为界)
                # 实际项目中可以使用 LangChain 的 MarkdownTextSplitter
                paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 10]
                
                for p in paragraphs:
                    documents.append(p)
                    metadatas.append({"source": filename})
                    ids.append(f"doc_{doc_id_counter}")
                    doc_id_counter += 1
                    
    # 4. 批量存入向量数据库
    if documents:
        print(f"准备将 {len(documents)} 个文本块写入向量数据库...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ 成功！向量数据库已保存至 {db_dir}")
    else:
        print("⚠️ 知识库中没有可用的文本内容。")

if __name__ == "__main__":
    build_vector_db()
