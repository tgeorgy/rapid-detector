// Canvas annotation system for bounding box drawing
class CanvasAnnotations {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.currentImage = null;
        this.userAnnotations = [];
        this.modelPredictions = [];
        this.scale = 1;
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        this.canvas.addEventListener('mousemove', this.draw.bind(this));
        this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));
        
        // Touch events for mobile
        this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));
    }

    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = this.canvas.getBoundingClientRect();
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                        e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
            clientX: touch.clientX - rect.left,
            clientY: touch.clientY - rect.top
        });
        this.canvas.dispatchEvent(mouseEvent);
    }

    loadImage(imageFile) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.currentImage = img;
                this.fitImageToCanvas(img);
                this.redraw();
                document.querySelector('.canvas-overlay').classList.add('hidden');
                resolve();
            };
            img.onerror = (error) => {
                reject(error);
            };

            if (imageFile instanceof File) {
                const reader = new FileReader();
                reader.onload = (e) => img.src = e.target.result;
                reader.onerror = (error) => reject(error);
                reader.readAsDataURL(imageFile);
            } else {
                img.src = imageFile; // URL string
            }
        });
    }

    fitImageToCanvas(img) {
        const maxWidth = 800;
        const maxHeight = 600;
        
        let { width, height } = img;
        
        // Calculate scale to fit image in canvas
        const scaleX = maxWidth / width;
        const scaleY = maxHeight / height;
        this.scale = Math.min(scaleX, scaleY);
        
        // Update canvas size to fit scaled image
        this.canvas.width = width * this.scale;
        this.canvas.height = height * this.scale;
        
        // Center canvas in container
        this.canvas.style.margin = '0 auto';
        this.canvas.style.display = 'block';
    }

    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();

        // Convert from screen coordinates to canvas buffer coordinates
        // (accounts for CSS scaling of the canvas element)
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const canvasX = (e.clientX - rect.left) * scaleX;
        const canvasY = (e.clientY - rect.top) * scaleY;

        // Convert from canvas buffer coordinates to original image coordinates
        return {
            x: canvasX / this.scale,
            y: canvasY / this.scale
        };
    }

    startDrawing(e) {
        if (!this.currentImage) return;
        
        this.isDrawing = true;
        const pos = this.getMousePos(e);
        this.startX = pos.x;
        this.startY = pos.y;
    }

    draw(e) {
        if (!this.isDrawing || !this.currentImage) return;
        
        const pos = this.getMousePos(e);
        this.redraw();
        
        // Get current annotation mode from detectorApp
        const isPositive = window.detectorApp ? window.detectorApp.currentAnnotationMode === 'positive' : true;
        const color = isPositive ? '#22c55e' : '#ef4444'; // Green for positive, red for negative
        
        
        // Draw current bounding box being created
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.strokeRect(
            this.startX * this.scale, 
            this.startY * this.scale,
            (pos.x - this.startX) * this.scale, 
            (pos.y - this.startY) * this.scale
        );
        this.ctx.setLineDash([]);
    }

    stopDrawing(e) {
        if (!this.isDrawing || !this.currentImage) return;
        
        this.isDrawing = false;
        const pos = this.getMousePos(e);
        
        // Only add annotation if box has meaningful size
        const width = Math.abs(pos.x - this.startX);
        const height = Math.abs(pos.y - this.startY);
        
        if (width > 10 && height > 10) {
            // Get current annotation mode from detectorApp
            const isPositive = window.detectorApp ? window.detectorApp.currentAnnotationMode === 'positive' : true;
            
            const annotation = {
                xmin: Math.min(this.startX, pos.x),
                ymin: Math.min(this.startY, pos.y),
                xmax: Math.max(this.startX, pos.x),
                ymax: Math.max(this.startY, pos.y),
                label: 'user_annotation',
                isPositive: isPositive
            };
            
            this.userAnnotations.push(annotation);
            this.redraw();
            
            // Dispatch custom event for annotation added
            window.dispatchEvent(new CustomEvent('annotationAdded', { 
                detail: annotation 
            }));
        }
    }

    addModelPredictions(predictions, idMaskPng = null) {
        this.modelPredictions = predictions.map(pred => ({
            xmin: pred.boxes?.[0] || pred.xmin || pred[0],
            ymin: pred.boxes?.[1] || pred.ymin || pred[1],
            xmax: pred.boxes?.[2] || pred.xmax || pred[2],
            ymax: pred.boxes?.[3] || pred.ymax || pred[3],
            confidence: pred.scores || pred.confidence || 0.5,
            label: pred.label || 'prediction'
        }));
        
        // Store ID mask for rendering
        this.idMaskPng = idMaskPng;
        
        // Load ID mask image if provided
        if (idMaskPng) {
            this.loadIdMask(idMaskPng);
        } else {
            this.redraw();
        }
    }

    loadIdMask(pngDataUrl) {
        if (!pngDataUrl) {
            this.redraw();
            return;
        }

        const maskImg = new Image();
        maskImg.onload = () => {
            this.idMaskImage = maskImg;
            this.redraw();
        };
        maskImg.onerror = (e) => {
            console.error('Failed to load ID mask image:', e);
            this.redraw();
        };
        maskImg.src = pngDataUrl;
    }

    getDetectionColor(detectionId) {
        // Expanded color palette with better distribution
        const colors = [
            '#FF4757', // Red
            '#2ED573', // Green  
            '#3742FA', // Blue
            '#FFA726', // Orange
            '#9C27B0', // Purple
            '#00ACC1', // Cyan
            '#FFD54F', // Yellow
            '#EF5350', // Light Red
            '#26A69A', // Teal
            '#AB47BC', // Light Purple
            '#42A5F5', // Light Blue
            '#66BB6A', // Light Green
            '#FF7043', // Deep Orange
            '#EC407A', // Pink
            '#5C6BC0', // Indigo
            '#26C6DA', // Light Cyan
            '#D4E157', // Lime
            '#FF8A65', // Deep Orange Light
            '#7E57C2', // Deep Purple
            '#29B6F6'  // Sky Blue
        ];
        
        // Use a hash-like function to distribute colors more randomly
        // This prevents adjacent IDs from getting similar colors
        const hash = (detectionId * 9973) % colors.length;
        return colors[hash];
    }

    renderIdMaskOverlay() {
        if (!this.idMaskImage || !this.currentImage) {
            return;
        }


        // Create an off-screen canvas to process the ID mask
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        // Set size to match ID mask
        tempCanvas.width = this.idMaskImage.width;
        tempCanvas.height = this.idMaskImage.height;
        
        // Draw ID mask to temp canvas to read pixel data
        tempCtx.drawImage(this.idMaskImage, 0, 0);
        
        // Get pixel data
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;
        
        // Check for non-zero pixels
        let nonZeroPixels = 0;
        let maxId = 0;
        for (let i = 0; i < data.length; i += 4) {
            const detectionId = data[i];
            if (detectionId > 0) {
                nonZeroPixels++;
                maxId = Math.max(maxId, detectionId);
            }
        }
        
        // Create colored overlay
        const overlayImageData = tempCtx.createImageData(tempCanvas.width, tempCanvas.height);
        const overlayData = overlayImageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            const detectionId = data[i]; // ID mask stores ID in red channel
            
            if (detectionId > 0) {
                // Get color for this detection ID
                const color = this.getDetectionColor(detectionId);
                const rgb = this.hexToRgb(color);
                
                overlayData[i] = rgb.r;     // Red
                overlayData[i + 1] = rgb.g; // Green
                overlayData[i + 2] = rgb.b; // Blue
                overlayData[i + 3] = 128;   // Alpha (0.5 transparency, 128/255 ≈ 0.5)
            } else {
                // Transparent for background
                overlayData[i + 3] = 0;
            }
        }
        
        // Put colored overlay back to temp canvas
        tempCtx.putImageData(overlayImageData, 0, 0);
        
        // Draw the colored mask overlay on main canvas, scaled to match image
        this.ctx.drawImage(
            tempCanvas,
            0, 0,
            this.currentImage.width * this.scale,
            this.currentImage.height * this.scale
        );
        
    }

    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : {r: 0, g: 255, b: 0}; // fallback to green
    }

    redraw() {
        if (!this.currentImage) return;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw image
        this.ctx.drawImage(
            this.currentImage, 
            0, 0, 
            this.currentImage.width * this.scale, 
            this.currentImage.height * this.scale
        );
        
        // Draw ID mask overlay if available
        if (this.idMaskImage) {
            this.renderIdMaskOverlay();
        } else {
            // Fallback: Draw model predictions as boxes (filled translucent green)
            this.modelPredictions.forEach(pred => {
                const x = pred.xmin * this.scale;
                const y = pred.ymin * this.scale;
                const w = (pred.xmax - pred.xmin) * this.scale;
                const h = (pred.ymax - pred.ymin) * this.scale;
                
                // Filled rectangle
                this.ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
                this.ctx.fillRect(x, y, w, h);
                
                // Border
                this.ctx.strokeStyle = '#00FF00';
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(x, y, w, h);
                
                // Label with confidence
                const label = `${pred.confidence.toFixed(2)}`;
                this.ctx.fillStyle = '#00FF00';
                this.ctx.fillRect(x, y - 20, this.ctx.measureText(label).width + 10, 20);
                this.ctx.fillStyle = 'black';
                this.ctx.font = '12px sans-serif';
                this.ctx.fillText(label, x + 5, y - 6);
            });
        }
        
        // Always draw bounding boxes and labels for predictions
        const showScores = document.getElementById('show-scores')?.checked ?? true;
        
        this.modelPredictions.forEach((pred, index) => {
            const x = pred.xmin * this.scale;
            const y = pred.ymin * this.scale;
            const w = (pred.xmax - pred.xmin) * this.scale;
            const h = (pred.ymax - pred.ymin) * this.scale;
            
            // Get detection color
            const color = this.getDetectionColor(index + 1);
            
            // Border (thinner and dashed)
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 1;
            this.ctx.setLineDash([4, 2]); // Dashed line pattern
            this.ctx.strokeRect(x, y, w, h);
            this.ctx.setLineDash([]); // Reset line dash
            
            // Label with confidence (only if toggle is enabled)
            if (showScores) {
                const label = `${pred.confidence.toFixed(2)}`;
                this.ctx.font = '12px sans-serif';
                const textWidth = this.ctx.measureText(label).width;
                
                // Translucent background rectangle
                const rgb = this.hexToRgb(color);
                this.ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.7)`;
                this.ctx.fillRect(x, y - 20, textWidth + 10, 20);
                
                // White text
                this.ctx.fillStyle = 'white';
                this.ctx.fillText(label, x + 5, y - 6);
            }
        });
        
        // Draw user annotations (red outlines)
        this.userAnnotations.forEach(ann => {
            const x = ann.xmin * this.scale;
            const y = ann.ymin * this.scale;
            const w = (ann.xmax - ann.xmin) * this.scale;
            const h = (ann.ymax - ann.ymin) * this.scale;
            
            // Use color based on annotation type (green for positive, red for negative)
            const color = ann.isPositive !== false ? '#22c55e' : '#ef4444';
            
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x, y, w, h);
            
            // Corner markers
            const markerSize = 6;
            this.ctx.fillStyle = color;
            this.ctx.fillRect(x - markerSize/2, y - markerSize/2, markerSize, markerSize);
            this.ctx.fillRect(x + w - markerSize/2, y - markerSize/2, markerSize, markerSize);
            this.ctx.fillRect(x - markerSize/2, y + h - markerSize/2, markerSize, markerSize);
            this.ctx.fillRect(x + w - markerSize/2, y + h - markerSize/2, markerSize, markerSize);
        });
    }

    clearAnnotations() {
        this.userAnnotations = [];
        this.redraw();
    }

    clearPredictions() {
        this.modelPredictions = [];
        this.idMaskImage = null;
        this.idMaskPng = null;
        this.redraw();
    }

    clearAll() {
        this.userAnnotations = [];
        this.modelPredictions = [];
        this.idMaskImage = null;
        this.idMaskPng = null;
        this.redraw();
    }

    getAnnotations() {
        return {
            userAnnotations: this.userAnnotations,
            modelPredictions: this.modelPredictions
        };
    }

    exportAnnotations() {
        if (!this.currentImage) return null;
        
        return {
            imageWidth: this.currentImage.width,
            imageHeight: this.currentImage.height,
            userAnnotations: this.userAnnotations.map(ann => ({
                ...ann,
                // Convert back to original image coordinates
                xmin: Math.round(ann.xmin),
                ymin: Math.round(ann.ymin),
                xmax: Math.round(ann.xmax),
                ymax: Math.round(ann.ymax)
            }))
        };
    }
}

// Initialize global canvas annotation system
let canvasAnnotations;

document.addEventListener('DOMContentLoaded', () => {
    canvasAnnotations = new CanvasAnnotations('annotation-canvas');
    
    // Add event listener for show scores toggle
    const showScoresCheckbox = document.getElementById('show-scores');
    if (showScoresCheckbox) {
        showScoresCheckbox.addEventListener('change', () => {
            canvasAnnotations.redraw();
        });
    }
});