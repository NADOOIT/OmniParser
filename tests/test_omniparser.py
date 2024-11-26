import unittest
import sys
import os
import torch
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import omniparser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omniparser import Omniparser, config

class TestOmniParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources"""
        cls.test_config = config.copy()
        cls.test_config.update({
            'batch_size': 16,
            'cache_models': True,
            'BOX_TRESHOLD': 0.05
        })
        cls.parser = Omniparser(cls.test_config)
        
        # Create test directory if it doesn't exist
        cls.test_dir = Path(__file__).parent / 'test_images'
        cls.test_dir.mkdir(exist_ok=True)

    def setUp(self):
        """Set up for each test"""
        self.test_image_path = str(self.test_dir / 'test_ui.png')
        # Create a simple test image if it doesn't exist
        if not os.path.exists(self.test_image_path):
            self._create_test_image()

    def _create_test_image(self):
        """Create a test image with known UI elements"""
        img = Image.new('RGB', (800, 600), color='white')
        # Add some UI elements for testing
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        # Draw a button
        draw.rectangle([100, 100, 200, 150], outline='black', fill='gray')
        draw.text((120, 115), "Button", fill='black')
        # Draw a viewport
        draw.rectangle([300, 100, 500, 400], outline='black')
        draw.text((350, 110), "Viewport", fill='black')
        img.save(self.test_image_path)

    def test_initialization(self):
        """Test parser initialization"""
        self.assertIsNotNone(self.parser)
        self.assertEqual(self.parser.config['batch_size'], 16)
        self.assertTrue(self.parser.config['cache_models'])

    def test_device_selection(self):
        """Test correct device selection"""
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(self.parser.device, expected_device)

    def test_basic_parsing(self):
        """Test basic image parsing functionality"""
        image, results = self.parser.parse(self.test_image_path)
        
        # Basic checks
        self.assertIsNotNone(image)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)

        # Check result structure
        for result in results:
            self.assertIn('from', result)
            self.assertIn('shape', result)
            self.assertIn('text', result)
            self.assertIn('type', result)
            
            # Check shape structure
            shape = result['shape']
            self.assertIn('x', shape)
            self.assertIn('y', shape)
            self.assertIn('width', shape)
            self.assertIn('height', shape)

    def test_element_detection(self):
        """Test detection of different UI elements"""
        image, results = self.parser.parse(self.test_image_path)
        
        # Check for different element types
        detected_types = set(result['type'] for result in results)
        
        # Should at least detect some basic types
        basic_types = {'text', 'button', 'viewport'}
        self.assertTrue(any(t in detected_types for t in basic_types))

    def test_caching(self):
        """Test model caching functionality"""
        # First parse
        start_time = time.time()
        self.parser.parse(self.test_image_path)
        first_parse_time = time.time() - start_time

        # Second parse (should be faster due to caching)
        start_time = time.time()
        self.parser.parse(self.test_image_path)
        second_parse_time = time.time() - start_time

        # Cache should make second parse faster
        self.assertLess(second_parse_time, first_parse_time)

    def test_batch_processing(self):
        """Test batch processing with different batch sizes"""
        # Test with different batch sizes
        batch_sizes = [8, 16, 32]
        times = []
        
        for batch_size in batch_sizes:
            self.parser.config['batch_size'] = batch_size
            start_time = time.time()
            self.parser.parse(self.test_image_path)
            times.append(time.time() - start_time)

        # Larger batches should generally be faster or similar
        # (not always true due to overhead, but generally trend should be there)
        self.assertTrue(any(times[i] <= times[i-1] * 1.5 for i in range(1, len(times))))

    def test_error_handling(self):
        """Test error handling for various scenarios"""
        # Test with non-existent image
        with self.assertRaises(Exception):
            self.parser.parse("nonexistent.png")

        # Test with invalid image
        invalid_image_path = str(self.test_dir / 'invalid.txt')
        with open(invalid_image_path, 'w') as f:
            f.write("Not an image")
        with self.assertRaises(Exception):
            self.parser.parse(invalid_image_path)

    def test_memory_cleanup(self):
        """Test memory cleanup on parser deletion"""
        parser = Omniparser(self.test_config)
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Use the parser
        parser.parse(self.test_image_path)
        
        # Delete the parser
        del parser
        
        # Check memory usage
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.assertLessEqual(final_memory, initial_memory * 1.1)  # Allow for some overhead

    def test_element_types_configuration(self):
        """Test configuration of element types"""
        # Test with limited element types
        limited_config = self.test_config.copy()
        limited_config['element_types'] = ['button']
        parser = Omniparser(limited_config)
        _, results = parser.parse(self.test_image_path)
        
        # Should only detect buttons and default types
        for result in results:
            self.assertIn(result['type'], ['button', 'text', 'icon'])

    def test_threshold_sensitivity(self):
        """Test sensitivity to different threshold settings"""
        thresholds = [0.01, 0.05, 0.1]
        results_count = []
        
        for threshold in thresholds:
            self.parser.config['BOX_TRESHOLD'] = threshold
            _, results = self.parser.parse(self.test_image_path)
            results_count.append(len(results))
        
        # Lower thresholds should generally detect more elements
        self.assertTrue(all(results_count[i] >= results_count[i+1] 
                          for i in range(len(results_count)-1)))

    def test_stress_test(self):
        """Test parser under heavy load"""
        # Create a large test image
        large_img = Image.new('RGB', (3840, 2160), color='white')  # 4K resolution
        draw = ImageDraw.Draw(large_img)
        
        # Add many UI elements
        for i in range(100):  # Add 100 elements
            x = (i % 10) * 350
            y = (i // 10) * 200
            draw.rectangle([x, y, x+300, y+150], outline='black', fill='gray')
            draw.text((x+20, y+20), f"Button {i}", fill='black')
        
        large_img_path = str(self.test_dir / 'stress_test.png')
        large_img.save(large_img_path)
        
        # Test parsing with different batch sizes
        batch_sizes = [16, 32, 64]
        for batch_size in batch_sizes:
            self.parser.config['batch_size'] = batch_size
            start_time = time.time()
            _, results = self.parser.parse(large_img_path)
            parse_time = time.time() - start_time
            
            # Verify reasonable processing time (adjust threshold as needed)
            self.assertLess(parse_time, 30.0, f"Parsing took too long with batch_size {batch_size}")
            # Verify detection count
            self.assertGreater(len(results), 50, f"Too few elements detected with batch_size {batch_size}")

    def test_edge_cases(self):
        """Test parser with edge cases"""
        # Test with tiny image
        tiny_img = Image.new('RGB', (50, 50), color='white')
        tiny_img_path = str(self.test_dir / 'tiny.png')
        tiny_img.save(tiny_img_path)
        _, results_tiny = self.parser.parse(tiny_img_path)
        
        # Test with single pixel image
        pixel_img = Image.new('RGB', (1, 1), color='white')
        pixel_img_path = str(self.test_dir / 'pixel.png')
        pixel_img.save(pixel_img_path)
        _, results_pixel = self.parser.parse(pixel_img_path)
        
        # Test with grayscale image
        gray_img = Image.new('L', (800, 600), color=128)
        gray_img_path = str(self.test_dir / 'gray.png')
        gray_img.save(gray_img_path)
        _, results_gray = self.parser.parse(gray_img_path)
        
        # Test with transparent image
        transparent_img = Image.new('RGBA', (800, 600), (255, 255, 255, 0))
        transparent_img_path = str(self.test_dir / 'transparent.png')
        transparent_img.save(transparent_img_path)
        _, results_transparent = self.parser.parse(transparent_img_path)

    def test_noise_robustness(self):
        """Test parser's robustness to noise"""
        # Create base image
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 200, 150], outline='black', fill='gray')
        draw.text((120, 115), "Button", fill='black')
        
        # Add random noise
        img_array = np.array(img)
        noise = np.random.normal(0, 25, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_img)
        
        noisy_img_path = str(self.test_dir / 'noisy.png')
        noisy_img.save(noisy_img_path)
        
        # Test parser with noisy image
        _, results = self.parser.parse(noisy_img_path)
        self.assertTrue(len(results) > 0, "Failed to detect elements in noisy image")

    def test_concurrent_processing(self):
        """Test parser with concurrent processing"""
        num_threads = 4
        num_iterations = 3
        results = []
        
        def process_image():
            return self.parser.parse(self.test_image_path)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_image) for _ in range(num_iterations)]
            results = [future.result() for future in futures]
        
        # Verify all results are consistent
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(len(result[1]), len(first_result[1]))

    def test_memory_leak(self):
        """Test for memory leaks during repeated processing"""
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            
            # Process multiple times
            for _ in range(10):
                self.parser.parse(self.test_image_path)
                torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            # Allow for small memory overhead
            self.assertLess(final_memory - initial_memory, 1024 * 1024 * 10)  # 10MB threshold

    def test_element_positions(self):
        """Test accuracy of element position detection"""
        # Create image with known element positions
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw elements at specific positions
        elements = [
            ('button', [100, 100, 200, 150], "Button 1"),
            ('viewport', [300, 200, 500, 400], "Viewport 1"),
            ('button', [600, 100, 700, 150], "Button 2")
        ]
        
        for element_type, bbox, text in elements:
            if element_type == 'button':
                draw.rectangle(bbox, outline='black', fill='gray')
            else:
                draw.rectangle(bbox, outline='black')
            draw.text((bbox[0]+10, bbox[1]+10), text, fill='black')
        
        test_img_path = str(self.test_dir / 'position_test.png')
        img.save(test_img_path)
        
        # Parse and verify positions
        _, results = self.parser.parse(test_img_path)
        
        # Verify each detected element is within reasonable bounds
        for result in results:
            shape = result['shape']
            self.assertGreaterEqual(shape['x'], 0)
            self.assertGreaterEqual(shape['y'], 0)
            self.assertLess(shape['x'] + shape['width'], 800)
            self.assertLess(shape['y'] + shape['height'], 600)

    def test_different_resolutions(self):
        """Test parser with different image resolutions"""
        resolutions = [
            (640, 480),    # VGA
            (1280, 720),   # HD
            (1920, 1080),  # Full HD
            (2560, 1440),  # 2K
            (3840, 2160)   # 4K
        ]
        
        for width, height in resolutions:
            # Create test image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Add scaled elements
            scale = min(width/800, height/600)
            button_size = int(100 * scale)
            draw.rectangle([100, 100, 100+button_size, 150+button_size], 
                         outline='black', fill='gray')
            
            img_path = str(self.test_dir / f'resolution_{width}x{height}.png')
            img.save(img_path)
            
            # Test parsing
            try:
                _, results = self.parser.parse(img_path)
                self.assertTrue(len(results) > 0, 
                              f"Failed to detect elements at resolution {width}x{height}")
            except Exception as e:
                self.fail(f"Failed to process {width}x{height} resolution: {str(e)}")

    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        # Track processing time
        start_time = time.time()
        _, results = self.parser.parse(self.test_image_path)
        process_time = time.time() - start_time
        
        # Basic performance assertions
        self.assertLess(process_time, 5.0, "Processing took too long")
        
        if torch.cuda.is_available():
            # Track GPU memory usage
            torch.cuda.reset_peak_memory_stats()
            _, results = self.parser.parse(self.test_image_path)
            peak_memory = torch.cuda.max_memory_allocated()
            
            # Memory usage should be reasonable
            self.assertLess(peak_memory, 1024 * 1024 * 1024)  # 1GB threshold

if __name__ == '__main__':
    unittest.main()
