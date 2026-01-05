// Main application logic
class DetectorApp {
    constructor() {
        this.uploadedImages = [];
        this.savedImageData = [];
        this.selectedImage = null;
        this.currentClassName = '';
        this.detectorName = '';
        this.apiBaseUrl = 'http://localhost:8000';
        this.examplesUpdateTimer = null;
        this.currentAnnotationMode = 'positive'; // Default to positive
        this.pendingAnnotations = []; // Store annotations before submission
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Image upload
        const imageUpload = document.getElementById('image-upload');
        const uploadArea = imageUpload.parentElement;
        
        imageUpload.addEventListener('change', this.handleImageUpload.bind(this));
        
        // Drag and drop
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        // Detector selection
        const detectorSelect = document.getElementById('detector-select');
        detectorSelect.addEventListener('change', this.handleDetectorSelection.bind(this));
        
        // Detector management buttons
        document.getElementById('new-detector-btn').addEventListener('click', this.showNewDetectorInput.bind(this));
        document.getElementById('delete-detector-btn').addEventListener('click', this.deleteDetector.bind(this));
        document.getElementById('confirm-new-btn').addEventListener('click', this.createNewDetector.bind(this));
        document.getElementById('cancel-new-btn').addEventListener('click', this.cancelNewDetector.bind(this));
        
        // New detector input
        const newClassNameInput = document.getElementById('new-class-name');
        newClassNameInput.addEventListener('keypress', this.handleNewDetectorKeypress.bind(this));
        
        // Annotation toolbar
        document.getElementById('positive-tool').addEventListener('click', () => this.setAnnotationMode('positive'));
        document.getElementById('negative-tool').addEventListener('click', () => this.setAnnotationMode('negative'));
        document.getElementById('clear-annotations').addEventListener('click', this.clearPendingAnnotations.bind(this));
        document.getElementById('submit-annotations').addEventListener('click', this.submitAnnotations.bind(this));
        
        // Buttons
        document.getElementById('save-detector-btn').addEventListener('click', this.saveDetector.bind(this));
        document.getElementById('show-api-btn').addEventListener('click', this.showAPIModal.bind(this));
        
        // Modal controls
        document.getElementById('copy-api-btn').addEventListener('click', this.copyAPICode.bind(this));
        document.getElementById('close-modal-btn').addEventListener('click', this.hideAPIModal.bind(this));
        document.querySelector('.modal-close').addEventListener('click', this.hideAPIModal.bind(this));
        document.getElementById('api-modal').addEventListener('click', (e) => {
            if (e.target.id === 'api-modal') this.hideAPIModal();
        });
        
        // Listen for annotations
        window.addEventListener('annotationAdded', this.onAnnotationAdded.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        e.target.closest('.upload-area').style.borderColor = '#667eea';
    }

    handleDrop(e) {
        e.preventDefault();
        const uploadArea = e.target.closest('.upload-area');
        uploadArea.style.borderColor = '#cbd5e0';
        
        const files = Array.from(e.dataTransfer.files).filter(file => 
            file.type.startsWith('image/')
        );
        
        if (files.length > 0) {
            this.processImages(files);
        }
    }

    handleImageUpload(e) {
        const files = Array.from(e.target.files);
        this.processImages(files);
    }

    processImages(files) {
        this.uploadedImages = files;
        this.updateImageGallery();
        this.updateStatus('✅ Images uploaded successfully');
        
        // Auto-select first image
        if (files.length > 0) {
            this.selectImage(0);
        }
    }

    updateImageGallery() {
        const gallery = document.getElementById('image-gallery');
        gallery.innerHTML = '';
        
        this.uploadedImages.forEach((file, index) => {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = `Image ${index + 1}`;
            img.className = 'gallery-image';
            img.addEventListener('click', () => this.selectImage(index));
            
            gallery.appendChild(img);
        });
    }

    selectImage(index) {
        // Clear pending annotations when switching images
        this.clearPendingAnnotations();
        
        // Update visual selection
        document.querySelectorAll('.gallery-image').forEach((img, i) => {
            img.classList.toggle('selected', i === index);
        });
        
        // Load image to canvas
        this.selectedImage = this.uploadedImages[index];
        canvasAnnotations.loadImage(this.selectedImage);
        
        this.updateStatus(`📸 Selected image ${index + 1}`);
        
        // Auto-run detection if we have a detector and class name
        if (this.currentClassName && this.detectorName) {
            this.updateStatus(`🔄 Auto-detecting on new image...`, 'info');
            setTimeout(() => this.runDetection(), 500); // Small delay to let image load
        }
    }

    async handleDetectorSelection(e) {
        const selectedDetector = e.target.value;
        this.currentClassName = selectedDetector;
        this.detectorName = selectedDetector;
        
        // Enable/disable delete button
        const deleteBtn = document.getElementById('delete-detector-btn');
        deleteBtn.disabled = !selectedDetector;
        
        // Clear pending annotations when switching detectors
        this.clearPendingAnnotations();
        
        // Load images for the selected detector
        if (selectedDetector) {
            await this.loadDetectorImages(selectedDetector);
            // Generate API code when selecting an existing detector
            this.generateAPICode();
        } else {
            // Clear images if no detector selected
            this.uploadedImages = [];
            this.savedImageData = [];
            this.selectedImage = null;
            this.updateImageGallery();
            // Clear API code when no detector selected
            document.getElementById('show-api-btn').style.display = 'none';
            this.apiCode = null;
        }
        
        // Update examples display
        this.updateExamplesDisplay();
        
        // Auto-run detection if we have a detector and selected image
        if (selectedDetector && this.selectedImage) {
            this.updateStatus(`🔄 Auto-detecting on selected detector...`, 'info');
            setTimeout(() => this.runDetection(), 500);
        }
    }

    showNewDetectorInput() {
        const newDetectorInput = document.getElementById('new-detector-input');
        const newClassNameInput = document.getElementById('new-class-name');
        
        newDetectorInput.style.display = 'block';
        newClassNameInput.focus();
    }

    handleNewDetectorKeypress(e) {
        if (e.key === 'Enter') {
            this.createNewDetector();
        } else if (e.key === 'Escape') {
            this.cancelNewDetector();
        }
    }

    async createNewDetector() {
        const newClassNameInput = document.getElementById('new-class-name');
        const className = newClassNameInput.value.trim();
        
        if (!className) {
            this.updateStatus('❌ Please enter a class name', 'error');
            return;
        }
        
        try {
            // Create new detector
            const response = await fetch(`${this.apiBaseUrl}/create_detector`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: className,
                    class_name: className,
                    is_semantic: true
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Reload detectors list and select the new one
            await this.loadDetectorsList();
            
            const detectorSelect = document.getElementById('detector-select');
            detectorSelect.value = className;
            this.handleDetectorSelection({ target: detectorSelect });
            
            this.updateStatus(`✅ Created new detector: ${className}`, 'success');
            this.cancelNewDetector();

        } catch (error) {
            console.error('Error creating detector:', error);
            this.updateStatus(`❌ Error creating detector: ${error.message}`, 'error');
        }
    }

    cancelNewDetector() {
        const newDetectorInput = document.getElementById('new-detector-input');
        const newClassNameInput = document.getElementById('new-class-name');
        
        newDetectorInput.style.display = 'none';
        newClassNameInput.value = '';
    }

    async deleteDetector() {
        if (!this.detectorName) {
            this.updateStatus('❌ No detector selected', 'error');
            return;
        }

        if (!confirm(`Are you sure you want to delete the detector "${this.detectorName}"? This action cannot be undone.`)) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/detect/${encodeURIComponent(this.detectorName)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Reload detectors list and clear selection
            await this.loadDetectorsList();
            
            const detectorSelect = document.getElementById('detector-select');
            detectorSelect.value = '';
            this.currentClassName = '';
            this.detectorName = '';
            
            // Disable delete button and update displays
            document.getElementById('delete-detector-btn').disabled = true;
            this.updateExamplesDisplay();
            
            this.updateStatus(`✅ Deleted detector successfully`, 'success');

        } catch (error) {
            console.error('Error deleting detector:', error);
            this.updateStatus(`❌ Error deleting detector: ${error.message}`, 'error');
        }
    }

    setAnnotationMode(mode) {
        this.currentAnnotationMode = mode;
        
        // Update toolbar visual state
        document.querySelectorAll('.tool-btn[data-type]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.type === mode);
        });
        
        this.updateStatus(`🎨 ${mode === 'positive' ? 'Positive' : 'Negative'} annotation mode active`);
    }

    onAnnotationAdded(e) {
        const annotation = e.detail;
        
        // Add annotation to pending list (annotation already has correct isPositive property)
        this.pendingAnnotations.push(annotation);
        
        // Update counter and UI
        this.updateAnnotationCounter();
        this.updateStatus(`🎯 Added ${this.currentAnnotationMode} annotation: ${Math.round(annotation.xmax - annotation.xmin)}x${Math.round(annotation.ymax - annotation.ymin)}px`);
    }

    updateAnnotationCounter() {
        const positiveCount = this.pendingAnnotations.filter(a => a.isPositive).length;
        const negativeCount = this.pendingAnnotations.filter(a => !a.isPositive).length;
        
        document.getElementById('positive-count').textContent = `✅ ${positiveCount}`;
        document.getElementById('negative-count').textContent = `❌ ${negativeCount}`;
        
        // Enable/disable submit button
        const submitBtn = document.getElementById('submit-annotations');
        submitBtn.disabled = this.pendingAnnotations.length === 0;
        
        if (this.pendingAnnotations.length > 0) {
            submitBtn.textContent = `Submit ${this.pendingAnnotations.length} Annotations`;
        } else {
            submitBtn.textContent = 'Submit Annotations';
        }
    }

    clearPendingAnnotations() {
        const hadAnnotations = this.pendingAnnotations.length > 0;
        
        this.pendingAnnotations = [];
        this.updateAnnotationCounter();
        
        // Clear canvas annotations (safely)
        if (typeof canvasAnnotations !== 'undefined' && canvasAnnotations.clearAnnotations) {
            canvasAnnotations.clearAnnotations();
        }
        
        if (hadAnnotations) {
            this.updateStatus('🗑️ Cleared all pending annotations');
        }
    }

    async submitAnnotations() {
        if (!this.currentClassName) {
            this.updateStatus('❌ Please enter a class name first', 'error');
            return;
        }

        if (!this.selectedImage) {
            this.updateStatus('❌ Please select an image first', 'error');
            return;
        }

        if (this.pendingAnnotations.length === 0) {
            this.updateStatus('❌ Please draw at least one bounding box', 'error');
            return;
        }

        try {
            this.updateStatus('📤 Submitting annotations...', 'info');
            
            // Group annotations by type for separate API calls
            const positiveAnnotations = this.pendingAnnotations.filter(a => a.isPositive);
            const negativeAnnotations = this.pendingAnnotations.filter(a => !a.isPositive);
            
            let totalSubmitted = 0;

            // Submit positive annotations
            if (positiveAnnotations.length > 0) {
                await this.submitAnnotationBatch(positiveAnnotations, true);
                totalSubmitted += positiveAnnotations.length;
            }

            // Submit negative annotations
            if (negativeAnnotations.length > 0) {
                await this.submitAnnotationBatch(negativeAnnotations, false);
                totalSubmitted += negativeAnnotations.length;
            }

            // Clear pending annotations and update UI
            this.pendingAnnotations = [];
            this.updateAnnotationCounter();
            canvasAnnotations.clearAnnotations();
            
            this.updateStatus(`✅ Submitted ${totalSubmitted} annotations successfully!`, 'success');
            
            // Update examples display
            this.updateExamplesDisplay();
            
            // Automatically run detection after successful annotation submission
            this.updateStatus('🔄 Running detection with new examples...', 'info');
            setTimeout(() => this.runDetection(), 500);

        } catch (error) {
            console.error('Error submitting annotations:', error);
            this.updateStatus(`❌ Error submitting annotations: ${error.message}`, 'error');
        }
    }

    async submitAnnotationBatch(annotations, isPositive) {
        // Prepare form data
        const formData = new FormData();
        formData.append('image', this.selectedImage);
        formData.append('class_name', this.currentClassName);
        formData.append('annotations', JSON.stringify(annotations));
        formData.append('is_positive', isPositive);

        const response = await fetch(`${this.apiBaseUrl}/add_example`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    async runDetection() {
        if (!this.currentClassName) {
            this.updateStatus('❌ Please enter a class name first', 'error');
            return;
        }

        if (!this.selectedImage) {
            this.updateStatus('❌ Please select an image first', 'error');
            return;
        }

        this.updateStatus('🔄 Running detection...', 'info');

        try {
            // Clear previous predictions
            canvasAnnotations.clearPredictions();

            // Prepare form data
            const formData = new FormData();
            formData.append('image', this.selectedImage);

            const response = await fetch(`${this.apiBaseUrl}/detect/${encodeURIComponent(this.detectorName)}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            
            // Process predictions
            const predictions = [];
            if (result.boxes && result.scores) {
                for (let i = 0; i < result.boxes.length; i++) {
                    predictions.push({
                        xmin: result.boxes[i][0],
                        ymin: result.boxes[i][1],
                        xmax: result.boxes[i][2],
                        ymax: result.boxes[i][3],
                        confidence: result.scores[i],
                        label: this.currentClassName
                    });
                }
            }

            // Display predictions on canvas (async for better performance)
            requestAnimationFrame(() => {
                canvasAnnotations.addModelPredictions(predictions, result.id_mask_png);
            });

            // Update status display
            if (predictions.length > 0) {
                const avgConf = predictions.length > 0 ? 
                    (predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length).toFixed(3) : '0';
                
                this.updateStatus(`🎯 Found ${predictions.length} detections (avg conf: ${avgConf}) - ${result.inference_time || 'N/A'}s`, 'success');
            } else {
                this.updateStatus('🔍 No objects detected', 'warning');
            }

        } catch (error) {
            console.error('Error running detection:', error);
            
            // Handle different error types more efficiently
            if (error.message.includes('404') || error.message.includes('not found')) {
                this.updateStatus('🔄 Creating new detector with text prompt...', 'info');
                await this.createTextOnlyDetector();
            } else if (error.message.includes('500')) {
                this.updateStatus(`❌ Server error during detection: ${error.message}`, 'error');
            } else {
                this.updateStatus(`❌ Detection error: ${error.message}`, 'error');
            }
        } finally {
        }
    }

    async createTextOnlyDetector() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/create_detector`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: this.detectorName,
                    class_name: this.currentClassName,
                    is_semantic: true
                })
            });

            if (response.ok) {
                this.updateStatus('✅ Text-only detector created. Try detection again.', 'success');
                // Automatically retry detection
                setTimeout(() => this.runDetection(), 1000);
            } else {
                throw new Error('Failed to create detector');
            }

        } catch (error) {
            this.updateStatus(`❌ Failed to create detector: ${error.message}`, 'error');
        }
    }

    async saveDetector() {
        if (!this.detectorName) {
            this.updateStatus('❌ Please enter a class name first', 'error');
            return;
        }

        try {
            // First save uploaded images if any
            if (this.uploadedImages.length > 0) {
                await this.saveUploadedImages();
            }

            const response = await fetch(`${this.apiBaseUrl}/save_detector/${encodeURIComponent(this.detectorName)}`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            
            // Generate API code
            this.generateAPICode();

            this.updateStatus('✅ Detector saved successfully!', 'success');

        } catch (error) {
            console.error('Error saving detector:', error);
            this.updateStatus(`❌ Error saving detector: ${error.message}`, 'error');
        }
    }

    async saveUploadedImages() {
        if (!this.detectorName || this.uploadedImages.length === 0) {
            return;
        }

        try {
            const formData = new FormData();
            this.uploadedImages.forEach((file, index) => {
                formData.append('images', file);
            });

            const response = await fetch(`${this.apiBaseUrl}/save_uploaded_images/${encodeURIComponent(this.detectorName)}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }


        } catch (error) {
            console.error('Error saving uploaded images:', error);
            this.updateStatus(`❌ Error saving images: ${error.message}`, 'error');
        }
    }

    async loadDetectorImages(detectorName) {
        try {
            this.updateStatus('📥 Loading saved images...', 'info');

            const response = await fetch(`${this.apiBaseUrl}/uploaded_images/${encodeURIComponent(detectorName)}`);
            
            if (!response.ok) {
                if (response.status === 404) {
                    // No saved images for this detector
                    this.uploadedImages = [];
                    this.savedImageData = [];
                    this.selectedImage = null;
                    this.updateImageGallery();
                    this.updateStatus(`No saved images for ${detectorName}`, 'info');
                    return;
                }
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            if (data.images && data.images.length > 0) {
                
                // Store the image data with URLs instead of base64 (much faster)
                this.savedImageData = data.images;
                this.uploadedImages = []; // Clear file objects temporarily
                
                // Update gallery to show saved images
                this.updateImageGalleryFromData();
                
                // Auto-select first image
                if (this.savedImageData.length > 0) {
                    this.selectSavedImage(0);
                }
                
                this.updateStatus(`📥 Loaded ${data.images.length} saved images`, 'success');
            } else {
                this.uploadedImages = [];
                this.savedImageData = [];
                this.selectedImage = null;
                this.updateImageGallery();
                this.updateStatus(`No images found for ${detectorName}`, 'info');
            }

        } catch (error) {
            console.error('Error loading detector images:', error);
            this.updateStatus(`❌ Error loading images: ${error.message}`, 'error');
        }
    }

    updateImageGalleryFromData() {
        const gallery = document.getElementById('image-gallery');
        gallery.innerHTML = '';
        
        this.savedImageData.forEach((imgData, index) => {
            const img = document.createElement('img');
            img.src = imgData.url; // Use static file URL directly
            img.alt = `Saved Image ${index + 1}`;
            img.className = 'gallery-image';
            img.addEventListener('click', () => this.selectSavedImage(index));
            
            gallery.appendChild(img);
        });
    }

    async selectSavedImage(index) {
        // Clear pending annotations when switching images
        this.clearPendingAnnotations();
        
        // Update visual selection
        document.querySelectorAll('.gallery-image').forEach((img, i) => {
            img.classList.toggle('selected', i === index);
        });
        
        const imgData = this.savedImageData[index];
        
        // Convert to File object only when needed for canvas (lazy conversion)
        const response = await fetch(imgData.url); // Use static file URL
        const blob = await response.blob();
        this.selectedImage = new File([blob], imgData.filename || `image_${index}.png`, { type: 'image/png' });
        
        canvasAnnotations.loadImage(this.selectedImage);
        
        this.updateStatus(`📸 Selected saved image ${index + 1}`);
        
        // Auto-run detection if we have a detector
        if (this.currentClassName && this.detectorName) {
            this.updateStatus(`🔄 Auto-detecting on saved image...`, 'info');
            setTimeout(() => this.runDetection(), 500);
        }
    }

    generateAPICode() {
        if (!this.detectorName) {
            document.getElementById('show-api-btn').style.display = 'none';
            return;
        }
        
        this.apiCode = `import requests

detector_name = "${this.detectorName}"
api_url = f"http://localhost:8000/detect/{detector_name}"

with open("test_image.jpg", "rb") as f:
    response = requests.post(api_url, files={"image": f})

results = response.json()
print(f"Found {len(results['boxes'])} detections")
for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
    print(f"Detection {i+1}: {box} (confidence: {score:.2f})")`;

        // Show the API button
        document.getElementById('show-api-btn').style.display = 'block';
    }
    
    showAPIModal() {
        if (!this.apiCode) {
            this.updateStatus('❌ No API code available. Please select a detector first.', 'error');
            return;
        }
        
        const codeElement = document.getElementById('api-code-modal');
        const modalElement = document.getElementById('api-modal');
        
        // Set the textarea value
        codeElement.value = this.apiCode;
        
        modalElement.classList.add('show');
        
        // Focus the textarea so Ctrl+A works immediately
        setTimeout(() => {
            codeElement.focus();
            codeElement.select(); // Select all text immediately
        }, 100);
    }
    
    hideAPIModal() {
        document.getElementById('api-modal').classList.remove('show');
    }
    
    async copyAPICode() {
        try {
            await navigator.clipboard.writeText(this.apiCode);
            this.updateStatus('✅ API code copied to clipboard!', 'success');
        } catch (err) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = this.apiCode;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.updateStatus('✅ API code copied to clipboard!', 'success');
        }
    }

    async updateExamplesDisplay() {
        if (!this.detectorName) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/detect/${encodeURIComponent(this.detectorName)}/examples`);
            
            if (!response.ok) {
                // If detector doesn't exist yet, show placeholder
                if (response.status === 404) {
                    this.showExamplesPlaceholder();
                    return;
                }
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.displayExamples(data.examples);

        } catch (error) {
            console.error('Error fetching examples:', error);
            this.showExamplesPlaceholder();
        }
    }

    showExamplesPlaceholder() {
        const examplesDiv = document.getElementById('object-examples');
        examplesDiv.innerHTML = `
            <div class="examples-placeholder">
                Add examples to see object cutouts here
            </div>
        `;
    }

    displayExamples(examples) {
        const examplesDiv = document.getElementById('object-examples');
        
        if (examples.length === 0) {
            this.showExamplesPlaceholder();
            return;
        }

        // Group examples by type
        const positiveExamples = examples.filter(ex => ex.is_positive);
        const negativeExamples = examples.filter(ex => !ex.is_positive);

        let html = '';
        
        if (positiveExamples.length > 0) {
            html += '<div class="example-group positive-examples">';
            html += '<h4>✅ Positive Examples</h4>';
            html += '<div class="example-grid">';
            positiveExamples.forEach(example => {
                html += `
                    <div class="example-item positive">
                        <img src="${example.cutout}" alt="Positive example" />
                        <div class="example-info">
                            <small>${Math.round(example.size[0])}×${Math.round(example.size[1])}</small>
                        </div>
                    </div>
                `;
            });
            html += '</div></div>';
        }

        if (negativeExamples.length > 0) {
            html += '<div class="example-group negative-examples">';
            html += '<h4>❌ Negative Examples</h4>';
            html += '<div class="example-grid">';
            negativeExamples.forEach(example => {
                html += `
                    <div class="example-item negative">
                        <img src="${example.cutout}" alt="Negative example" />
                        <div class="example-info">
                            <small>${Math.round(example.size[0])}×${Math.round(example.size[1])}</small>
                        </div>
                    </div>
                `;
            });
            html += '</div></div>';
        }

        examplesDiv.innerHTML = html;
    }

    updateStatus(message, type = 'info') {
        // Skip redundant messages
        if (this.lastMessage === message && Date.now() - this.lastMessageTime < 2000) {
            return;
        }
        
        this.lastMessage = message;
        this.lastMessageTime = Date.now();
        
        this.showToast(message, type);
    }

    showToast(message, type = 'info', duration = 2500) {
        const container = document.getElementById('toast-container');
        
        // Limit number of visible toasts
        const existingToasts = container.querySelectorAll('.toast');
        if (existingToasts.length >= 3) {
            this.removeToast(existingToasts[0]);
        }
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const closeBtn = document.createElement('button');
        closeBtn.className = 'toast-close';
        closeBtn.innerHTML = '×';
        closeBtn.onclick = () => this.removeToast(toast);
        
        const messageSpan = document.createElement('span');
        messageSpan.textContent = message;
        
        toast.appendChild(messageSpan);
        toast.appendChild(closeBtn);
        container.appendChild(toast);
        
        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 10);
        
        // Shorter auto-dismiss times based on type
        let autoRemoveDelay = duration;
        if (type === 'success') autoRemoveDelay = 1500;
        if (type === 'info') autoRemoveDelay = 2000;
        if (type === 'warning') autoRemoveDelay = 3000;
        if (type === 'error') autoRemoveDelay = 5000;
        
        setTimeout(() => this.removeToast(toast), autoRemoveDelay);
    }

    removeToast(toast) {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 200); // Match CSS transition duration
    }

    async loadDetectorsList() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/detectors`);
            if (response.ok) {
                const data = await response.json();
                const detectors = data.detectors || [];
                
                // Update dropdown
                const detectorSelect = document.getElementById('detector-select');
                const currentValue = detectorSelect.value;
                
                // Clear existing options except the first one
                detectorSelect.innerHTML = '<option value="">-- Select or create detector --</option>';
                
                // Add detectors to dropdown
                detectors.forEach(detectorName => {
                    const option = document.createElement('option');
                    option.value = detectorName;
                    option.textContent = detectorName;
                    detectorSelect.appendChild(option);
                });
                
                // Restore selection if it still exists and trigger load
                if (currentValue && detectors.includes(currentValue)) {
                    detectorSelect.value = currentValue;
                    // Manually trigger the selection to load images and examples
                    await this.handleDetectorSelection({ target: detectorSelect });
                }
                
            } else {
            }
        } catch (error) {
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.detectorApp = new DetectorApp();
    
    // Initialize annotation counter
    detectorApp.updateAnnotationCounter();
    
    // Load available detectors
    detectorApp.loadDetectorsList();
    
});