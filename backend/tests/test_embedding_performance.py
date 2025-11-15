"""
Performance and benchmark tests for the embedding system
Tests speed, memory usage, and scalability
"""

import pytest
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import psutil
import os
from memory_profiler import memory_usage
import concurrent.futures
from threading import Thread
import queue

from app.services.rag.embeddings import embedding_service


class TestEmbeddingPerformance:
    """Test performance characteristics of embedding system"""

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        """Generate sample texts for performance testing"""
        return [
            f"Financial report {i}: The company showed strong performance in Q{i%4 + 1} with revenue growth of {i%20 + 5}% year-over-year."
            for i in range(100)
        ]

    @pytest.fixture
    def large_text(self) -> str:
        """Generate a large text document"""
        # Approximately 5000 words
        base_text = "The financial performance of the company has been exceptional. "
        return base_text * 250

    def test_single_embedding_speed(self):
        """Test speed of single text embedding"""
        text = "Apple reported strong quarterly earnings with revenue up 15%"

        # Warm-up
        _ = embedding_service.embed_text(text)

        # Measure time for multiple runs
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = embedding_service.embed_text(text)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\nSingle embedding performance:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Std Dev: {std_time*1000:.2f}ms")
        print(f"  Min: {min(times)*1000:.2f}ms")
        print(f"  Max: {max(times)*1000:.2f}ms")

        # Assert reasonable performance (< 100ms average)
        assert avg_time < 0.1, f"Single embedding too slow: {avg_time:.3f}s"

    def test_batch_embedding_speed(self, sample_texts):
        """Test speed of batch embeddings"""
        batch_sizes = [10, 25, 50, 100]
        results = []

        for size in batch_sizes:
            texts = sample_texts[:size]

            # Warm-up
            _ = embedding_service.embed_texts(texts, show_progress=False)

            # Measure
            start = time.perf_counter()
            embeddings = embedding_service.embed_texts(texts, show_progress=False)
            elapsed = time.perf_counter() - start

            throughput = size / elapsed
            results.append({
                'batch_size': size,
                'time': elapsed,
                'throughput': throughput,
                'per_item': elapsed / size
            })

            print(f"\nBatch size {size}:")
            print(f"  Total time: {elapsed:.2f}s")
            print(f"  Throughput: {throughput:.1f} texts/sec")
            print(f"  Per item: {elapsed/size*1000:.2f}ms")

        # Verify batch processing is more efficient than individual
        assert results[-1]['per_item'] < results[0]['per_item'] * 0.8

    def test_memory_usage(self, sample_texts):
        """Test memory consumption during embedding"""

        def embed_batch():
            return embedding_service.embed_texts(sample_texts[:50], show_progress=False)

        # Measure memory usage
        mem_usage = memory_usage(embed_batch, interval=0.1, timeout=30)

        if mem_usage:
            baseline = mem_usage[0]
            peak = max(mem_usage)
            memory_increase = peak - baseline

            print(f"\nMemory usage:")
            print(f"  Baseline: {baseline:.1f} MB")
            print(f"  Peak: {peak:.1f} MB")
            print(f"  Increase: {memory_increase:.1f} MB")

            # Assert reasonable memory usage (< 500MB increase for 50 texts)
            assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f} MB"

    def test_concurrent_embedding_requests(self):
        """Test concurrent embedding generation"""
        texts = [
            f"Concurrent test document {i}: Financial analysis and insights"
            for i in range(20)
        ]

        def embed_text(text):
            return embedding_service.embed_text(text)

        # Sequential baseline
        start = time.perf_counter()
        sequential_results = [embed_text(t) for t in texts]
        sequential_time = time.perf_counter() - start

        # Concurrent execution
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(embed_text, texts))
        concurrent_time = time.perf_counter() - start

        print(f"\nConcurrency test:")
        print(f"  Sequential: {sequential_time:.2f}s")
        print(f"  Concurrent (4 workers): {concurrent_time:.2f}s")
        print(f"  Speedup: {sequential_time/concurrent_time:.2f}x")

        # Verify results are consistent
        for seq, con in zip(sequential_results, concurrent_results):
            similarity = np.dot(seq, con) / (np.linalg.norm(seq) * np.linalg.norm(con))
            assert similarity > 0.999, "Concurrent results differ from sequential"

    def test_large_text_handling(self, large_text):
        """Test performance with very large texts"""
        # Test different text sizes
        text_sizes = [1000, 5000, 10000, 50000]  # character counts
        results = []

        for size in text_sizes:
            text = large_text[:size]

            start = time.perf_counter()
            embedding = embedding_service.embed_text(text)
            elapsed = time.perf_counter() - start

            results.append({
                'size': size,
                'time': elapsed,
                'chars_per_sec': size / elapsed
            })

            print(f"\nText size {size} chars:")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Speed: {size/elapsed:.0f} chars/sec")

            # Verify embedding dimension is consistent
            assert len(embedding) == 768

    def test_model_loading_time(self):
        """Test model initialization/loading time"""
        from app.services.rag.embeddings import EmbeddingService

        # Create new instance (forces model reload)
        start = time.perf_counter()
        service = EmbeddingService()
        load_time = time.perf_counter() - start

        print(f"\nModel loading time: {load_time:.2f}s")

        # First embedding (includes any lazy initialization)
        start = time.perf_counter()
        _ = service.embed_text("Test text")
        first_embed_time = time.perf_counter() - start

        print(f"First embedding time: {first_embed_time:.3f}s")

        # Subsequent embedding (warmed up)
        start = time.perf_counter()
        _ = service.embed_text("Test text")
        second_embed_time = time.perf_counter() - start

        print(f"Second embedding time: {second_embed_time:.3f}s")

        # Second should be much faster than first
        assert second_embed_time < first_embed_time * 0.5

    def test_embedding_cache_behavior(self):
        """Test if embeddings are cached or regenerated"""
        text = "Test caching behavior of embeddings"

        # Generate embedding multiple times
        times = []
        embeddings = []

        for i in range(5):
            start = time.perf_counter()
            emb = embedding_service.embed_text(text)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            embeddings.append(emb)

        # Check if times are consistent (no caching would show similar times)
        time_variance = np.std(times) / np.mean(times)
        print(f"\nCache behavior test:")
        print(f"  Time variance: {time_variance:.3f}")
        print(f"  Times: {[f'{t*1000:.1f}ms' for t in times]}")

        # Verify embeddings are identical
        for i in range(1, len(embeddings)):
            similarity = np.dot(embeddings[0], embeddings[i]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[i])
            )
            assert similarity > 0.999999, "Same text produces different embeddings"


class TestEmbeddingScalability:
    """Test scalability of the embedding system"""

    def test_incremental_load(self):
        """Test system behavior under increasing load"""
        load_levels = [10, 50, 100, 200, 500]
        results = []

        for num_texts in load_levels:
            texts = [
                f"Document {i}: Financial performance metrics and analysis"
                for i in range(num_texts)
            ]

            # Measure
            start = time.perf_counter()
            embeddings = embedding_service.embed_texts(texts, show_progress=False)
            elapsed = time.perf_counter() - start

            results.append({
                'num_texts': num_texts,
                'total_time': elapsed,
                'per_text': elapsed / num_texts,
                'throughput': num_texts / elapsed
            })

            print(f"\nLoad level {num_texts} texts:")
            print(f"  Total: {elapsed:.2f}s")
            print(f"  Per text: {elapsed/num_texts*1000:.1f}ms")
            print(f"  Throughput: {num_texts/elapsed:.1f} texts/sec")

        # Check that per-item time doesn't degrade significantly
        first_per_item = results[0]['per_text']
        last_per_item = results[-1]['per_text']
        degradation = (last_per_item - first_per_item) / first_per_item

        print(f"\nScalability:")
        print(f"  Per-item degradation: {degradation*100:.1f}%")

        # Assert degradation is less than 50%
        assert degradation < 0.5, f"Significant performance degradation: {degradation*100:.1f}%"

    def test_stress_test(self):
        """Stress test with continuous high load"""
        duration = 10  # seconds
        texts_processed = 0
        errors = []

        def process_continuously():
            nonlocal texts_processed, errors
            end_time = time.time() + duration

            while time.time() < end_time:
                try:
                    text = f"Stress test document {texts_processed}"
                    _ = embedding_service.embed_text(text)
                    texts_processed += 1
                except Exception as e:
                    errors.append(str(e))

        # Run stress test
        start = time.time()
        process_continuously()
        actual_duration = time.time() - start

        throughput = texts_processed / actual_duration

        print(f"\nStress test results:")
        print(f"  Duration: {actual_duration:.1f}s")
        print(f"  Texts processed: {texts_processed}")
        print(f"  Throughput: {throughput:.1f} texts/sec")
        print(f"  Errors: {len(errors)}")

        # Assert no errors and reasonable throughput
        assert len(errors) == 0, f"Errors during stress test: {errors[:5]}"
        assert throughput > 5, f"Low throughput under stress: {throughput:.1f} texts/sec"

    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation"""
        import gc

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Process many batches
        for batch_num in range(10):
            texts = [
                f"Batch {batch_num} document {i}: Testing for memory leaks"
                for i in range(50)
            ]
            _ = embedding_service.embed_texts(texts, show_progress=False)

            if batch_num % 3 == 0:
                gc.collect()

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        print(f"\nMemory leak test:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")

        # Assert memory increase is reasonable (< 100MB for 500 texts)
        assert memory_increase < 100, f"Potential memory leak: {memory_increase:.1f} MB increase"


class TestEmbeddingBenchmarks:
    """Comparative benchmarks for the embedding system"""

    def test_financial_domain_benchmark(self):
        """Benchmark on financial domain-specific texts"""
        financial_texts = {
            'earnings': [
                "Q3 earnings exceeded expectations with EPS of $1.25",
                "Revenue grew 15% YoY driven by cloud services",
                "Operating margins expanded 200 basis points"
            ],
            'risk': [
                "Market volatility poses significant downside risk",
                "Regulatory changes may impact profitability",
                "Credit risk exposure increased in emerging markets"
            ],
            'guidance': [
                "Full-year guidance raised to $5.00-$5.20 EPS",
                "Expecting 20% revenue growth in FY2025",
                "Capital allocation focused on high-ROI projects"
            ]
        }

        results = {}
        total_texts = sum(len(texts) for texts in financial_texts.values())

        # Benchmark each category
        for category, texts in financial_texts.items():
            start = time.perf_counter()
            embeddings = embedding_service.embed_texts(texts, show_progress=False)
            elapsed = time.perf_counter() - start

            results[category] = {
                'count': len(texts),
                'time': elapsed,
                'per_text': elapsed / len(texts)
            }

        # Overall benchmark
        all_texts = [text for texts in financial_texts.values() for text in texts]
        start = time.perf_counter()
        all_embeddings = embedding_service.embed_texts(all_texts, show_progress=False)
        total_time = time.perf_counter() - start

        print("\nFinancial Domain Benchmark:")
        for category, stats in results.items():
            print(f"  {category}: {stats['time']:.3f}s for {stats['count']} texts")
        print(f"  Total: {total_time:.3f}s for {total_texts} texts")
        print(f"  Average: {total_time/total_texts*1000:.1f}ms per text")

        # Verify all embeddings generated successfully
        assert len(all_embeddings) == total_texts
        assert all(len(emb) == 768 for emb in all_embeddings)

    def test_comparison_single_vs_batch(self):
        """Compare single vs batch processing efficiency"""
        texts = [
            f"Comparison test document {i}: Financial metrics and KPIs"
            for i in range(50)
        ]

        # Single processing
        single_start = time.perf_counter()
        single_embeddings = []
        for text in texts:
            emb = embedding_service.embed_text(text)
            single_embeddings.append(emb)
        single_time = time.perf_counter() - single_start

        # Batch processing
        batch_start = time.perf_counter()
        batch_embeddings = embedding_service.embed_texts(texts, show_progress=False)
        batch_time = time.perf_counter() - batch_start

        speedup = single_time / batch_time

        print(f"\nSingle vs Batch Comparison:")
        print(f"  Single (50 texts): {single_time:.2f}s")
        print(f"  Batch (50 texts): {batch_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency gain: {(1 - batch_time/single_time)*100:.1f}%")

        # Verify batch is significantly faster
        assert speedup > 1.5, f"Insufficient batch processing speedup: {speedup:.2f}x"

        # Verify results are identical
        for single, batch in zip(single_embeddings, batch_embeddings):
            similarity = np.dot(single, batch) / (np.linalg.norm(single) * np.linalg.norm(batch))
            assert similarity > 0.999999

    def test_real_world_document_sizes(self):
        """Test with realistic financial document sizes"""
        document_types = {
            'tweet': 50,          # ~50 chars (short social media)
            'headline': 100,      # ~100 chars (news headline)
            'abstract': 500,      # ~500 chars (executive summary)
            'paragraph': 1000,    # ~1000 chars (typical paragraph)
            'section': 5000,      # ~5000 chars (document section)
            'full_doc': 20000     # ~20000 chars (full document)
        }

        results = []

        for doc_type, char_count in document_types.items():
            # Generate text of appropriate size
            base = "Financial performance analysis. "
            text = base * (char_count // len(base))
            text = text[:char_count]

            # Measure embedding time
            start = time.perf_counter()
            embedding = embedding_service.embed_text(text)
            elapsed = time.perf_counter() - start

            results.append({
                'type': doc_type,
                'size': char_count,
                'time': elapsed,
                'chars_per_sec': char_count / elapsed if elapsed > 0 else 0
            })

            print(f"\n{doc_type} ({char_count} chars):")
            print(f"  Time: {elapsed*1000:.1f}ms")
            print(f"  Speed: {char_count/elapsed if elapsed > 0 else 0:.0f} chars/sec")

        # All should complete successfully
        assert all(r['time'] > 0 for r in results)