#!/usr/bin/env python3
"""
FinSearch AI Complete RAG CLI
End-to-end financial Q&A with retrieval, reranking, and LLM generation
"""

import os
import sys

# Add scripts directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, List, Dict, Any
import argparse
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

console = Console()

class FinSearchRAG:
    """Complete RAG pipeline with LLM generation"""

    def __init__(self,
                 use_ollama: bool = True,
                 use_openai: bool = False,
                 use_context: bool = True,
                 company_filter: Optional[str] = None):
        """Initialize RAG components and LLM"""
        self.use_context = use_context
        self.company_filter = company_filter
        self.llm_service = None
        self.llm_type = None

        console.print("[cyan]Initializing FinSearch RAG System...[/cyan]")

        # Initialize retrieval components
        self._init_retrieval()

        # Initialize LLM
        if use_ollama:
            self._init_ollama()
        elif use_openai:
            self._init_openai()
        else:
            console.print("[yellow]Warning: No LLM configured, will only show retrieved context[/yellow]")

    def _init_retrieval(self):
        """Initialize retrieval and reranking components"""
        try:
            # Connect to ChromaDB
            self.chroma_client = chromadb.PersistentClient(path='data/chroma_db')
            self.collection = self.chroma_client.get_collection('financial_documents')

            # Initialize embedding model (matching the database)
            self.embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')

            # Initialize reranker
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

            doc_count = self.collection.count()
            console.print(f"[green]✓ Connected to vector database ({doc_count:,} documents)[/green]")
            console.print("[green]✓ Initialized embedder and reranker[/green]")

        except Exception as e:
            console.print(f"[red]✗ Failed to initialize retrieval: {e}[/red]")
            sys.exit(1)

    def _init_ollama(self):
        """Initialize Ollama for local LLM generation"""
        try:
            import requests
            # Check if Ollama is running
            response = requests.get('http://localhost:11434/api/tags')
            if response.status_code == 200:
                models = response.json().get('models', [])

                # Look for gemma3 model first, then qwen, then any available model
                selected_model = None
                for model in models:
                    model_name = model['name'].lower()
                    # Prioritize gemma3:270m for fast generation
                    if 'gemma3:270m' in model['name']:
                        selected_model = model['name']
                        break
                    elif 'gemma' in model_name and not selected_model:
                        selected_model = model['name']
                    elif 'qwen' in model_name and not selected_model:
                        selected_model = model['name']

                if selected_model:
                    self.llm_service = 'ollama'
                    self.llm_type = selected_model
                    console.print(f"[green]✓ Connected to Ollama (Model: {selected_model})[/green]")
                elif models:
                    model_name = models[0]['name']
                    self.llm_service = 'ollama'
                    self.llm_type = model_name
                    console.print(f"[green]✓ Connected to Ollama (Model: {model_name})[/green]")
                else:
                    console.print("[yellow]Warning: Ollama running but no models found[/yellow]")
                    console.print("[yellow]Pull a model first: ollama pull gemma3:270m[/yellow]")
            else:
                raise Exception("Ollama not responding")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not connect to Ollama: {e}[/yellow]")
            console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")
            console.print("[yellow]Continuing without LLM generation...[/yellow]")

    def _init_openai(self):
        """Initialize OpenAI API"""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise Exception("OPENAI_API_KEY not set")

            openai.api_key = api_key
            self.llm_service = 'openai'
            self.llm_type = 'gpt-3.5-turbo'
            console.print(f"[green]✓ Connected to OpenAI (Model: {self.llm_type})[/green]")

        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize OpenAI: {e}[/yellow]")
            console.print("[yellow]Set OPENAI_API_KEY environment variable[/yellow]")

    def retrieve_context(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Retrieve and rerank relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()

            # Build filter
            where_filter = {}
            if self.company_filter:
                where_filter = {"ticker": self.company_filter}

            # Retrieve from ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * 2,  # Get more for reranking
                where=where_filter if where_filter else None
            )

            if not results['documents'][0]:
                return []

            # Prepare for reranking
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]

            # Rerank
            pairs = [[query, doc] for doc in documents]
            scores = self.reranker.predict(pairs)

            # Sort by score and return top results
            ranked_results = []
            for idx, score in enumerate(scores):
                ranked_results.append({
                    'text': documents[idx],
                    'score': float(score),
                    'metadata': metadatas[idx]
                })

            ranked_results.sort(key=lambda x: x['score'], reverse=True)
            return ranked_results[:n_results]

        except Exception as e:
            console.print(f"[red]Retrieval error: {e}[/red]")
            return []

    def generate_with_ollama(self, query: str, context: str) -> str:
        """Generate response using Ollama"""
        try:
            import requests
            import json

            prompt = f"""You are a financial analyst assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer: """

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.llm_type,
                    'prompt': prompt,
                    'stream': False
                }
            )

            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return "Error generating response from Ollama"

        except Exception as e:
            return f"Error: {e}"

    def generate_with_openai(self, query: str, context: str) -> str:
        """Generate response using OpenAI"""
        try:
            import openai

            messages = [
                {"role": "system", "content": "You are a financial analyst assistant. Answer questions based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]

            response = openai.ChatCompletion.create(
                model=self.llm_type,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message['content']

        except Exception as e:
            return f"Error: {e}"

    def get_response(self, query: str):
        """Get complete RAG response"""
        contexts = []

        # Step 1: Retrieve context if enabled
        if self.use_context:
            contexts = self.retrieve_context(query, n_results=5)

            if not contexts:
                console.print("[yellow]No relevant context found[/yellow]")
                context_text = ""
            else:
                # Combine top contexts
                context_texts = []
                for i, ctx in enumerate(contexts[:3], 1):
                    context_texts.append(f"[Document {i}]\n{ctx['text'][:500]}")
                context_text = "\n\n".join(context_texts)
        else:
            context_text = ""

        # Step 2: Generate response
        if self.llm_service == 'ollama':
            response = self.generate_with_ollama(query, context_text)
        elif self.llm_service == 'openai':
            response = self.generate_with_openai(query, context_text)
        elif context_text:
            # No LLM, just show context
            response = "**Retrieved Context:**\n\n" + context_text
        else:
            response = "No LLM configured and no context retrieved."

        return response, contexts

    def run_interactive(self):
        """Run interactive chat session"""
        console.print(Panel.fit(
            "[bold cyan]FinSearch AI - Complete RAG System[/bold cyan]\n"
            f"Mode: {'RAG-Enhanced' if self.use_context else 'Direct LLM'}\n"
            f"LLM: {self.llm_type or 'None (Context Only)'}\n"
            f"Company Filter: {self.company_filter or 'All Companies'}\n"
            "Type 'exit' or 'quit' to end session\n"
            "Type '/help' for commands",
            title="Welcome"
        ))

        while True:
            try:
                # Get user input
                query = Prompt.ask("\n[bold green]You[/bold green]")

                # Check for exit commands
                if query.lower() in ['exit', 'quit', 'q']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                # Check for special commands
                if query == '/help':
                    self.show_help()
                    continue
                elif query == '/toggle':
                    self.use_context = not self.use_context
                    console.print(f"[cyan]Context mode: {'ON' if self.use_context else 'OFF'}[/cyan]")
                    continue
                elif query.startswith('/company'):
                    parts = query.split()
                    if len(parts) > 1:
                        self.company_filter = parts[1].upper()
                        console.print(f"[cyan]Company filter set to: {self.company_filter}[/cyan]")
                    else:
                        self.company_filter = None
                        console.print("[cyan]Company filter cleared[/cyan]")
                    continue
                elif query == '/stats':
                    self.show_stats()
                    continue

                # Get and display response
                console.print("\n[bold blue]FinSearch AI[/bold blue] [dim](thinking...)[/dim]")
                response, contexts = self.get_response(query)

                # Clear thinking message and show response
                console.print("\033[1A\033[2K", end='')  # Clear previous line
                console.print("[bold blue]FinSearch AI:[/bold blue]")

                # Display response with markdown formatting
                console.print(Markdown(response))

                # Show sources if available
                if contexts and self.use_context:
                    console.print("\n[dim]Sources:[/dim]")
                    for i, ctx in enumerate(contexts[:3], 1):
                        metadata = ctx.get('metadata', {})
                        ticker = metadata.get('ticker', 'Unknown')
                        doc_type = metadata.get('doc_type', 'Unknown')
                        score = ctx.get('score', 0)
                        console.print(f"  [{i}] {ticker} - {doc_type} (score: {score:.2f})", style="dim")

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def show_help(self):
        """Display help information"""
        help_text = """
[bold]Available Commands:[/bold]
  /help          - Show this help message
  /toggle        - Toggle context/RAG mode on/off
  /company TICK  - Filter by company ticker (e.g., /company AAPL)
  /company       - Clear company filter
  /stats         - Show database statistics
  exit/quit      - Exit the CLI

[bold]Example Questions:[/bold]
  • What was Apple's revenue in 2024?
  • What are Amazon's main risk factors?
  • Compare Microsoft and Google's operating margins
  • What is Tesla's net income for Q3 2024?
  • Explain Apple's Services segment growth
        """
        console.print(Panel(help_text, title="Help"))

    def show_stats(self):
        """Show database statistics"""
        try:
            count = self.collection.count()

            # Get sample to show companies
            sample = self.collection.get(limit=1000)
            companies = set()
            doc_types = set()

            for metadata in sample.get('metadatas', []):
                if 'ticker' in metadata:
                    companies.add(metadata['ticker'])
                if 'doc_type' in metadata:
                    doc_types.add(metadata['doc_type'])

            stats_text = f"""
[bold]Database Statistics:[/bold]
  Total Documents: {count:,}
  Companies: {', '.join(sorted(companies)[:10])}{'...' if len(companies) > 10 else ''}
  Document Types: {', '.join(sorted(doc_types))}
            """
            console.print(Panel(stats_text, title="Stats"))

        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/red]")


def main():
    """Main entry point"""
    # Set environment variable for PyTorch usage
    os.environ['USE_TORCH'] = '1'

    parser = argparse.ArgumentParser(description='FinSearch AI Complete RAG CLI')
    parser.add_argument('--no-context', action='store_true',
                       help='Disable RAG context retrieval')
    parser.add_argument('--company', type=str,
                       help='Filter by company ticker (e.g., AAPL)')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI instead of Ollama')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM generation (context only)')
    parser.add_argument('query', nargs='*',
                       help='Direct query (non-interactive mode)')

    args = parser.parse_args()

    # Initialize CLI
    cli = FinSearchRAG(
        use_ollama=not args.use_openai and not args.no_llm,
        use_openai=args.use_openai,
        use_context=not args.no_context,
        company_filter=args.company.upper() if args.company else None
    )

    if args.query:
        # Non-interactive mode - single query
        query = ' '.join(args.query)
        response, contexts = cli.get_response(query)
        console.print(Markdown(response))

        if contexts and not args.no_context:
            console.print("\n[dim]Sources:[/dim]")
            for ctx in contexts[:3]:
                metadata = ctx.get('metadata', {})
                ticker = metadata.get('ticker', 'Unknown')
                doc_type = metadata.get('doc_type', 'Unknown')
                console.print(f"  • {ticker} - {doc_type}", style="dim")
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()