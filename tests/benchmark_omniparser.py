import sys
import os
import time
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import itertools

# Add parent directory to path to import omniparser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omniparser import Omniparser, config

class OmniParserBenchmark:
    def __init__(self, base_config=None):
        self.base_config = base_config if base_config else config.copy()
        self.results = {}
        self.parser = Omniparser(self.base_config)
        
        # Create test directory if it doesn't exist
        self.test_dir = Path(__file__).parent / 'test_images'
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test images if they don't exist
        self.create_test_suite()

    def create_test_suite(self):
        """Create a suite of test images with different characteristics"""
        # Simple UI
        self._create_test_image('simple_ui.png', 800, 600, elements=[
            ('button', [100, 100, 200, 150], "Submit"),
            ('viewport', [300, 100, 500, 400], "Main View")
        ])
        
        # Complex UI
        self._create_test_image('complex_ui.png', 1024, 768, elements=[
            ('button', [100, 100, 200, 150], "Submit"),
            ('viewport', [300, 100, 500, 400], "Main View"),
            ('dropdown', [600, 100, 700, 150], "Select"),
            ('checkbox', [100, 200, 120, 220], "Check"),
            ('slider', [200, 300, 400, 320], "Volume"),
            ('textbox', [500, 500, 700, 550], "Enter text...")
        ])
        
        # High Resolution UI
        self._create_test_image('hires_ui.png', 1920, 1080, elements=[
            ('button', [100, 100, 200, 150], "Submit"),
            ('viewport', [300, 100, 500, 400], "Main View"),
            ('button', [600, 100, 700, 150], "Cancel"),
            ('viewport', [800, 100, 1000, 400], "Side Panel")
        ])

    def _create_test_image(self, name, width, height, elements):
        """Create a test image with specified elements"""
        img_path = self.test_dir / name
        if not img_path.exists():
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            for element_type, bbox, text in elements:
                if element_type == 'button':
                    draw.rectangle(bbox, outline='black', fill='gray')
                elif element_type == 'viewport':
                    draw.rectangle(bbox, outline='black')
                elif element_type == 'dropdown':
                    draw.rectangle(bbox, outline='black', fill='lightgray')
                    draw.polygon([(bbox[2]-20, bbox[1]+10), (bbox[2]-10, bbox[3]-10), (bbox[2]-30, bbox[3]-10)], fill='black')
                elif element_type == 'checkbox':
                    draw.rectangle(bbox, outline='black')
                elif element_type == 'slider':
                    draw.rectangle(bbox, outline='black', fill='lightgray')
                    draw.rectangle([bbox[0]+50, bbox[1], bbox[0]+70, bbox[3]], fill='gray')
                elif element_type == 'textbox':
                    draw.rectangle(bbox, outline='black')
                
                # Add text label
                text_bbox = draw.textbbox((0, 0), text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = bbox[0] + (bbox[2] - bbox[0] - text_width) // 2
                text_y = bbox[1] + (bbox[3] - bbox[1] - text_height) // 2
                draw.text((text_x, text_y), text, fill='black')
            
            img.save(img_path)

    def benchmark_real_world_scenarios(self):
        """Benchmark parser with real-world UI scenarios"""
        scenarios = {
            'login_form': [
                ('textbox', [100, 100, 300, 130], "Username"),
                ('textbox', [100, 150, 300, 180], "Password"),
                ('button', [100, 200, 200, 230], "Login"),
                ('checkbox', [100, 250, 120, 270], "Remember me"),
                ('link', [250, 250, 350, 270], "Forgot password?")
            ],
            'dashboard': [
                ('viewport', [50, 50, 250, 550], "Navigation"),
                ('button', [70, 70, 230, 100], "Home"),
                ('button', [70, 120, 230, 150], "Profile"),
                ('button', [70, 170, 230, 200], "Settings"),
                ('viewport', [300, 50, 950, 550], "Main Content"),
                ('chart', [350, 100, 550, 300], "Statistics"),
                ('table', [600, 100, 900, 300], "Data Table"),
                ('button', [350, 350, 450, 380], "Export"),
                ('dropdown', [500, 350, 650, 380], "Filter")
            ],
            'media_player': [
                ('viewport', [50, 50, 750, 450], "Video Player"),
                ('slider', [50, 470, 750, 490], "Progress"),
                ('button', [50, 500, 80, 530], "Play"),
                ('button', [90, 500, 120, 530], "Pause"),
                ('slider', [650, 500, 750, 530], "Volume"),
                ('button', [130, 500, 160, 530], "Fullscreen")
            ]
        }
        
        results = {}
        for scenario_name, elements in scenarios.items():
            # Create scenario image
            img = Image.new('RGB', (1000, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            for element_type, bbox, text in elements:
                if element_type in ['button', 'textbox']:
                    draw.rectangle(bbox, outline='black', fill='lightgray')
                elif element_type == 'viewport':
                    draw.rectangle(bbox, outline='black')
                elif element_type == 'checkbox':
                    draw.rectangle(bbox, outline='black')
                    draw.line([bbox[0]+2, bbox[1]+2, bbox[2]-2, bbox[3]-2], fill='black')
                elif element_type == 'slider':
                    draw.rectangle(bbox, outline='black', fill='lightgray')
                    handle_x = bbox[0] + (bbox[2] - bbox[0]) // 2
                    draw.rectangle([handle_x-5, bbox[1]-2, handle_x+5, bbox[3]+2], fill='gray')
                
                # Add text label
                text_bbox = draw.textbbox((0, 0), text)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = bbox[0] + (bbox[2] - bbox[0] - text_width) // 2
                text_y = bbox[1] - 20
                draw.text((text_x, text_y), text, fill='black')
            
            img_path = str(self.test_dir / f'scenario_{scenario_name}.png')
            img.save(img_path)
            
            # Benchmark scenario
            times = []
            detections = []
            memory_usage = []
            
            for _ in range(5):  # Run multiple times for stability
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                _, detected = self.parser.parse(img_path)
                times.append(time.time() - start_time)
                detections.append(len(detected))
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated())
            
            results[scenario_name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_detections': np.mean(detections),
                'avg_memory': np.mean(memory_usage) if memory_usage else None,
                'expected_elements': len(elements),
                'detection_rate': np.mean(detections) / len(elements)
            }
        
        self.results['real_world'] = results
        return results

    def benchmark_batch_sizes(self, batch_sizes=[8, 16, 32, 64]):
        """Benchmark different batch sizes"""
        results = {}
        for batch_size in batch_sizes:
            config = self.base_config.copy()
            config['batch_size'] = batch_size
            parser = Omniparser(config)
            
            times = []
            for img_name in ['simple_ui.png', 'complex_ui.png', 'hires_ui.png']:
                start_time = time.time()
                parser.parse(str(self.test_dir / img_name))
                times.append(time.time() - start_time)
            
            results[batch_size] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'times': times
            }
        
        self.results['batch_size'] = results
        return results

    def benchmark_thresholds(self, thresholds=[0.01, 0.03, 0.05, 0.07, 0.1]):
        """Benchmark different detection thresholds"""
        results = {}
        for threshold in thresholds:
            config = self.base_config.copy()
            config['BOX_TRESHOLD'] = threshold
            parser = Omniparser(config)
            
            detections = []
            times = []
            for img_name in ['simple_ui.png', 'complex_ui.png', 'hires_ui.png']:
                start_time = time.time()
                _, results_list = parser.parse(str(self.test_dir / img_name))
                times.append(time.time() - start_time)
                detections.append(len(results_list))
            
            results[threshold] = {
                'avg_detections': np.mean(detections),
                'avg_time': np.mean(times),
                'detections': detections,
                'times': times
            }
        
        self.results['threshold'] = results
        return results

    def benchmark_caching(self, num_iterations=5):
        """Benchmark caching performance"""
        results = {
            'with_cache': {'times': []},
            'without_cache': {'times': []}
        }
        
        # Test with caching
        config = self.base_config.copy()
        config['cache_models'] = True
        parser = Omniparser(config)
        
        for _ in range(num_iterations):
            start_time = time.time()
            parser.parse(str(self.test_dir / 'complex_ui.png'))
            results['with_cache']['times'].append(time.time() - start_time)
        
        # Test without caching
        config['cache_models'] = False
        parser = Omniparser(config)
        
        for _ in range(num_iterations):
            start_time = time.time()
            parser.parse(str(self.test_dir / 'complex_ui.png'))
            results['without_cache']['times'].append(time.time() - start_time)
        
        self.results['caching'] = results
        return results

    def benchmark_adversarial_cases(self):
        """Benchmark parser with adversarial cases"""
        cases = {
            'overlapping': self._create_overlapping_elements(),
            'nested': self._create_nested_elements(),
            'dense': self._create_dense_elements(),
            'sparse': self._create_sparse_elements(),
            'low_contrast': self._create_low_contrast_elements()
        }
        
        results = {}
        for case_name, img_path in cases.items():
            # Benchmark with different thresholds
            thresholds = [0.01, 0.05, 0.1]
            case_results = []
            
            for threshold in thresholds:
                self.parser.config['BOX_TRESHOLD'] = threshold
                
                start_time = time.time()
                _, detected = self.parser.parse(img_path)
                
                case_results.append({
                    'threshold': threshold,
                    'time': time.time() - start_time,
                    'detections': len(detected)
                })
            
            results[case_name] = case_results
        
        self.results['adversarial'] = results
        return results

    def _create_overlapping_elements(self):
        """Create test image with overlapping elements"""
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Create overlapping buttons
        positions = [(100, 100), (150, 150), (200, 200)]
        for i, (x, y) in enumerate(positions):
            draw.rectangle([x, y, x+100, y+50], outline='black', fill=f'rgb({50*i}, {50*i}, {50*i})')
            draw.text((x+10, y+10), f"Button {i+1}", fill='white')
        
        img_path = str(self.test_dir / 'overlapping.png')
        img.save(img_path)
        return img_path

    def _create_nested_elements(self):
        """Create test image with nested elements"""
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Create nested viewports
        sizes = [(400, 400), (300, 300), (200, 200), (100, 100)]
        for i, (w, h) in enumerate(sizes):
            x = (800 - w) // 2
            y = (600 - h) // 2
            draw.rectangle([x, y, x+w, y+h], outline='black')
            draw.text((x+10, y+10), f"Level {i+1}", fill='black')
        
        img_path = str(self.test_dir / 'nested.png')
        img.save(img_path)
        return img_path

    def _create_dense_elements(self):
        """Create test image with densely packed elements"""
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Create grid of small buttons
        for i in range(20):
            for j in range(15):
                x = i * 40
                y = j * 40
                draw.rectangle([x, y, x+35, y+35], outline='black', fill='gray')
        
        img_path = str(self.test_dir / 'dense.png')
        img.save(img_path)
        return img_path

    def _create_sparse_elements(self):
        """Create test image with sparsely distributed elements"""
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Create few scattered elements
        positions = [(50, 50), (700, 50), (50, 500), (700, 500)]
        for i, (x, y) in enumerate(positions):
            draw.rectangle([x, y, x+50, y+30], outline='black', fill='gray')
            draw.text((x+10, y+10), str(i+1), fill='black')
        
        img_path = str(self.test_dir / 'sparse.png')
        img.save(img_path)
        return img_path

    def _create_low_contrast_elements(self):
        """Create test image with low contrast elements"""
        img = Image.new('RGB', (800, 600), color='rgb(240, 240, 240)')
        draw = ImageDraw.Draw(img)
        
        # Create low contrast elements
        for i in range(5):
            x = 100 + i * 120
            y = 250
            color = f'rgb({245-i*5}, {245-i*5}, {245-i*5})'
            draw.rectangle([x, y, x+100, y+50], outline=color, fill=color)
            draw.text((x+10, y+10), f"Button {i+1}", fill='rgb(250, 250, 250)')
        
        img_path = str(self.test_dir / 'low_contrast.png')
        img.save(img_path)
        return img_path

    def find_optimal_settings(self):
        """Find optimal settings based on benchmarks"""
        # Run all benchmarks if not already run
        if not self.results:
            self.benchmark_real_world_scenarios()
            self.benchmark_batch_sizes()
            self.benchmark_thresholds()
            self.benchmark_caching()
            self.benchmark_adversarial_cases()
        
        # Analyze results
        optimal_settings = {}
        
        # Find optimal batch size (best speed-memory tradeoff)
        batch_results = self.results['batch_size']
        optimal_batch = min(batch_results.keys(), 
                          key=lambda x: batch_results[x]['avg_time'])
        optimal_settings['batch_size'] = optimal_batch
        
        # Find optimal threshold (balance between speed and detection quality)
        threshold_results = self.results['threshold']
        optimal_threshold = min(threshold_results.keys(),
                              key=lambda x: threshold_results[x]['avg_time'] * 
                                          (1 + abs(threshold_results[x]['avg_detections'] - 10)))
        optimal_settings['BOX_TRESHOLD'] = optimal_threshold
        
        # Determine if caching is beneficial
        cache_results = self.results['caching']
        cache_benefit = (np.mean(cache_results['without_cache']['times']) - 
                        np.mean(cache_results['with_cache']['times']))
        optimal_settings['cache_models'] = cache_benefit > 0
        
        return optimal_settings

    def plot_results(self, save_dir=None):
        """Plot benchmark results"""
        if not save_dir:
            save_dir = self.test_dir / 'benchmark_results'
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Plot batch size results
        if 'batch_size' in self.results:
            plt.figure(figsize=(10, 6))
            batch_sizes = list(self.results['batch_size'].keys())
            times = [r['avg_time'] for r in self.results['batch_size'].values()]
            plt.plot(batch_sizes, times, 'o-')
            plt.xlabel('Batch Size')
            plt.ylabel('Average Processing Time (s)')
            plt.title('Batch Size vs Processing Time')
            plt.savefig(save_dir / 'batch_size_benchmark.png')
            plt.close()
        
        # Plot threshold results
        if 'threshold' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            thresholds = list(self.results['threshold'].keys())
            times = [r['avg_time'] for r in self.results['threshold'].values()]
            detections = [r['avg_detections'] for r in self.results['threshold'].values()]
            
            ax1.plot(thresholds, times, 'o-')
            ax1.set_xlabel('Detection Threshold')
            ax1.set_ylabel('Average Processing Time (s)')
            ax1.set_title('Threshold vs Processing Time')
            
            ax2.plot(thresholds, detections, 'o-')
            ax2.set_xlabel('Detection Threshold')
            ax2.set_ylabel('Average Number of Detections')
            ax2.set_title('Threshold vs Detection Count')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'threshold_benchmark.png')
            plt.close()
        
        # Plot caching results
        if 'caching' in self.results:
            plt.figure(figsize=(10, 6))
            labels = ['With Cache', 'Without Cache']
            times = [
                np.mean(self.results['caching']['with_cache']['times']),
                np.mean(self.results['caching']['without_cache']['times'])
            ]
            plt.bar(labels, times)
            plt.ylabel('Average Processing Time (s)')
            plt.title('Cache Performance Comparison')
            plt.savefig(save_dir / 'cache_benchmark.png')
            plt.close()

        # Plot real-world scenario results
        if 'real_world' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            scenarios = list(self.results['real_world'].keys())
            times = [r['avg_time'] for r in self.results['real_world'].values()]
            detection_rates = [r['detection_rate'] for r in self.results['real_world'].values()]
            
            ax1.plot(scenarios, times, 'o-')
            ax1.set_xlabel('Scenario')
            ax1.set_ylabel('Average Processing Time (s)')
            ax1.set_title('Scenario vs Processing Time')
            
            ax2.plot(scenarios, detection_rates, 'o-')
            ax2.set_xlabel('Scenario')
            ax2.set_ylabel('Detection Rate')
            ax2.set_title('Scenario vs Detection Rate')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'real_world_benchmark.png')
            plt.close()

        # Plot adversarial case results
        if 'adversarial' in self.results:
            fig, ax = plt.subplots(figsize=(10, 6))
            cases = list(self.results['adversarial'].keys())
            times = [np.mean([r['time'] for r in self.results['adversarial'][case]]) for case in cases]
            ax.plot(cases, times, 'o-')
            ax.set_xlabel('Adversarial Case')
            ax.set_ylabel('Average Processing Time (s)')
            ax.set_title('Adversarial Case vs Processing Time')
            plt.tight_layout()
            plt.savefig(save_dir / 'adversarial_benchmark.png')
            plt.close()

    def save_results(self, filename='benchmark_results.json'):
        """Save benchmark results to file"""
        save_path = self.test_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=4)

def main():
    # Run benchmarks
    benchmark = OmniParserBenchmark()
    
    print("Running real-world scenario benchmark...")
    benchmark.benchmark_real_world_scenarios()
    
    print("Running batch size benchmark...")
    benchmark.benchmark_batch_sizes()
    
    print("Running threshold benchmark...")
    benchmark.benchmark_thresholds()
    
    print("Running cache benchmark...")
    benchmark.benchmark_caching()
    
    print("Running adversarial case benchmark...")
    benchmark.benchmark_adversarial_cases()
    
    # Find optimal settings
    optimal_settings = benchmark.find_optimal_settings()
    print("\nOptimal settings found:")
    print(json.dumps(optimal_settings, indent=2))
    
    # Plot and save results
    print("\nGenerating plots...")
    benchmark.plot_results()
    benchmark.save_results()
    
    print("\nBenchmark results have been saved to the test_images directory")

if __name__ == '__main__':
    main()
