/**
 * SOTA Enrichment UI Module
 *
 * Provides streaming enrichment with preview/approval workflow
 */

export class EnrichmentManager {
    constructor() {
        this.activeEnrichments = new Map();  // conversation_id -> EventSource
        this.previewData = new Map();  // conversation_id -> preview data
    }

    /**
     * Enrich a single conversation with streaming updates
     */
    async enrichSingle(conversationId, fields = ['title', 'summary', 'tags', 'embedding'], options = {}) {
        const {
            onProgress = () => {},
            onComplete = () => {},
            onError = () => {}
        } = options;

        try {
            // Close existing stream if any
            this.closeStream(conversationId);

            // Open EventSource for streaming
            const eventSource = new EventSource(
                `/api/enrich/single?conversation_id=${conversationId}&fields=${fields.join(',')}`
            );

            this.activeEnrichments.set(conversationId, eventSource);

            const partialResults = {
                title: null,
                summary: null,
                tags: null,
                embedding: null
            };

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'partial') {
                    // Streaming update
                    partialResults[data.field] = data.value;
                    onProgress({
                        field: data.field,
                        value: data.value,
                        partial: true,
                        confidence: data.confidence
                    });

                } else if (data.type === 'final') {
                    // Field complete
                    partialResults[data.field] = data.value;
                    onProgress({
                        field: data.field,
                        value: data.value,
                        partial: false,
                        confidence: data.confidence,
                        cost: data.cost
                    });

                } else if (data.type === 'complete') {
                    // All enrichment complete
                    eventSource.close();
                    this.activeEnrichments.delete(conversationId);

                    onComplete({
                        results: partialResults,
                        totalCost: data.total_cost
                    });

                } else if (data.type === 'error') {
                    eventSource.close();
                    this.activeEnrichments.delete(conversationId);
                    onError(new Error(data.message));
                }
            };

            eventSource.onerror = (error) => {
                eventSource.close();
                this.activeEnrichments.delete(conversationId);
                onError(error);
            };

        } catch (error) {
            onError(error);
        }
    }

    /**
     * Bulk enrich multiple conversations
     */
    async enrichBulk(conversationIds, fields = ['title', 'summary', 'tags'], options = {}) {
        const {
            onProgress = () => {},
            onComplete = () => {},
            onError = () => {}
        } = options;

        try {
            const response = await fetch('/api/enrich/bulk', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    conversation_ids: conversationIds,
                    fields: fields
                })
            });

            if (!response.ok) {
                throw new Error(`Bulk enrichment failed: ${response.statusText}`);
            }

            // Read SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const {done, value} = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, {stream: true});
                const lines = buffer.split('\n');
                buffer = lines.pop();  // Keep incomplete line

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'progress') {
                            onProgress({
                                current: data.current,
                                total: data.total,
                                conversationId: data.conversation_id,
                                percent: data.percent
                            });

                        } else if (data.type === 'conversation_complete') {
                            onProgress({
                                type: 'conversation_done',
                                conversationId: data.conversation_id,
                                cost: data.cost
                            });

                        } else if (data.type === 'bulk_complete') {
                            onComplete({
                                totalProcessed: data.total_processed,
                                totalCost: data.total_cost
                            });

                        } else if (data.type === 'error') {
                            onError(new Error(data.message));
                        }
                    }
                }
            }

        } catch (error) {
            onError(error);
        }
    }

    /**
     * Get preview data for a conversation
     */
    async getPreview(conversationId) {
        const response = await fetch(`/api/enrich/preview/${conversationId}`);

        if (!response.ok) {
            if (response.status === 404) {
                return null;  // No preview available
            }
            throw new Error(`Failed to get preview: ${response.statusText}`);
        }

        const preview = await response.json();
        this.previewData.set(conversationId, preview);
        return preview;
    }

    /**
     * Approve and persist enrichment
     */
    async approveEnrichment(conversationId) {
        const response = await fetch('/api/enrich/approve', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({conversation_id: conversationId})
        });

        if (!response.ok) {
            throw new Error(`Failed to approve enrichment: ${response.statusText}`);
        }

        this.previewData.delete(conversationId);
        return await response.json();
    }

    /**
     * Approve enrichment with edited values
     */
    async approveEnrichmentWithEdits(conversationId, editedValues) {
        const response = await fetch('/api/enrich/approve', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                conversation_id: conversationId,
                edited_values: editedValues
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to approve enrichment: ${response.statusText}`);
        }

        this.previewData.delete(conversationId);
        return await response.json();
    }

    /**
     * Reject and discard enrichment
     */
    async rejectEnrichment(conversationId) {
        const response = await fetch('/api/enrich/reject', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({conversation_id: conversationId})
        });

        if (!response.ok) {
            throw new Error(`Failed to reject enrichment: ${response.statusText}`);
        }

        this.previewData.delete(conversationId);
        return await response.json();
    }

    /**
     * Estimate cost for enrichment
     */
    async estimateCost(conversationIds, fields) {
        const response = await fetch('/api/enrich/estimate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                conversation_ids: conversationIds,
                fields: fields
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to estimate cost: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Check health of enrichment system
     */
    async checkHealth() {
        const response = await fetch('/api/enrich/health');

        if (!response.ok) {
            throw new Error(`Health check failed: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Close streaming connection
     */
    closeStream(conversationId) {
        const eventSource = this.activeEnrichments.get(conversationId);
        if (eventSource) {
            eventSource.close();
            this.activeEnrichments.delete(conversationId);
        }
    }

    /**
     * Close all active streams
     */
    closeAllStreams() {
        for (const [id, eventSource] of this.activeEnrichments) {
            eventSource.close();
        }
        this.activeEnrichments.clear();
    }
}

/**
 * UI Component: Enrichment Button for conversation cards
 */
export class EnrichmentButton {
    constructor(conversationId, container, manager) {
        this.conversationId = conversationId;
        this.container = container;
        this.manager = manager;
        this.button = null;
        this.render();
    }

    render() {
        this.button = document.createElement('button');
        this.button.className = 'enrich-btn';
        this.button.innerHTML = '✨ Enrich';
        this.button.title = 'Enrich with SOTA models';

        this.button.onclick = () => this.handleClick();

        this.container.appendChild(this.button);
    }

    async handleClick() {
        try {
            // Disable button
            this.button.disabled = true;
            this.button.innerHTML = '⏳ Enriching...';

            // Start enrichment
            await this.manager.enrichSingle(
                this.conversationId,
                ['title', 'summary', 'tags'],
                {
                    onProgress: (update) => {
                        if (!update.partial) {
                            console.log(`Enriched ${update.field}: ${update.value}`);
                        }
                    },
                    onComplete: (result) => {
                        this.button.innerHTML = '✓ Enriched';
                        this.button.classList.add('enriched');

                        // Show preview modal
                        this.showPreviewModal(result);
                    },
                    onError: (error) => {
                        console.error('Enrichment failed:', error);
                        this.button.innerHTML = '❌ Failed';
                        this.button.classList.add('error');
                        alert(`Enrichment failed: ${error.message}`);
                    }
                }
            );

        } catch (error) {
            console.error('Enrichment error:', error);
            this.button.innerHTML = '❌ Error';
            this.button.disabled = false;
        }
    }

    async showPreviewModal(result) {
        // Get preview data
        const preview = await this.manager.getPreview(this.conversationId);

        if (!preview) {
            alert('Preview not available');
            return;
        }

        // Create modal
        const modal = new EnrichmentPreviewModal(
            this.conversationId,
            preview,
            this.manager
        );

        modal.show();
    }
}

/**
 * UI Component: Enrichment Preview Modal with Diff View
 */
export class EnrichmentPreviewModal {
    constructor(conversationId, preview, manager) {
        this.conversationId = conversationId;
        this.preview = preview;
        this.manager = manager;
        this.modal = null;
        this.editMode = false;
        this.editedValues = {};  // Store edited values
    }

    show() {
        // Create modal overlay
        this.modal = document.createElement('div');
        this.modal.className = 'enrichment-modal-overlay';

        this.modal.innerHTML = `
            <div class="enrichment-modal">
                <div class="modal-header">
                    <h2>✨ Enrichment Preview</h2>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <button class="btn btn-secondary edit-toggle-btn" style="padding: 6px 12px;">
                            <span class="edit-icon">✏️</span> Edit
                        </button>
                        <button class="close-btn" onclick="this.closest('.enrichment-modal-overlay').remove()">×</button>
                    </div>
                </div>

                <div class="modal-body">
                    ${this.renderDiff()}
                </div>

                <div class="modal-footer">
                    <div class="cost-info">
                        <span class="cost-label">Cost:</span>
                        <span class="cost-value">$${this.preview.total_cost.toFixed(4)}</span>
                    </div>
                    <div class="action-buttons">
                        <button class="btn btn-secondary reject-btn">Discard</button>
                        <button class="btn btn-primary approve-btn">Save Changes</button>
                    </div>
                </div>
            </div>
        `;

        // Add event listeners
        const approveBtn = this.modal.querySelector('.approve-btn');
        const rejectBtn = this.modal.querySelector('.reject-btn');
        const editToggleBtn = this.modal.querySelector('.edit-toggle-btn');

        approveBtn.onclick = () => this.approve();
        rejectBtn.onclick = () => this.reject();
        editToggleBtn.onclick = () => this.toggleEditMode();

        // Show modal
        document.body.appendChild(this.modal);
    }

    renderDiff() {
        let html = '<div class="enrichment-diff">';

        // Show changes for each field
        for (const result of this.preview.results) {
            if (result.field === 'embedding') continue;  // Skip embedding display

            const fieldValue = this.editedValues[result.field] !== undefined
                ? this.editedValues[result.field]
                : result.value;

            html += `
                <div class="diff-section">
                    <h3>${this.formatFieldName(result.field)}</h3>
                    <div class="diff-comparison">
                        <div class="diff-original">
                            <div class="diff-label">Original:</div>
                            <div class="diff-content">${this.formatValue(result.original_value)}</div>
                        </div>
                        <div class="diff-arrow">→</div>
                        <div class="diff-new">
                            <div class="diff-label">New (${(result.confidence * 100).toFixed(0)}% confidence):</div>
                            <div class="diff-content enriched" data-field="${result.field}">
                                ${this.renderEditableValue(result.field, fieldValue)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        html += '</div>';
        return html;
    }

    renderEditableValue(field, value) {
        if (Array.isArray(value)) {
            // For tags, render as comma-separated editable text
            return `<div class="editable-field" contenteditable="false" data-field="${field}">${this.escapeHtml(value.join(', '))}</div>`;
        } else {
            // For text fields (title, summary)
            return `<div class="editable-field" contenteditable="false" data-field="${field}">${this.escapeHtml(String(value))}</div>`;
        }
    }

    formatFieldName(field) {
        return field.charAt(0).toUpperCase() + field.slice(1);
    }

    formatValue(value) {
        if (value === null || value === undefined || value === '') {
            return '<em class="empty-value">(empty)</em>';
        }

        if (Array.isArray(value)) {
            return value.map(v => `<span class="tag">${this.escapeHtml(v)}</span>`).join(' ');
        }

        return this.escapeHtml(String(value));
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    toggleEditMode() {
        this.editMode = !this.editMode;

        const editToggleBtn = this.modal.querySelector('.edit-toggle-btn');
        const editableFields = this.modal.querySelectorAll('.editable-field');

        if (this.editMode) {
            // Enable editing
            editToggleBtn.innerHTML = '<span class="edit-icon">💾</span> Done Editing';
            editToggleBtn.classList.remove('btn-secondary');
            editToggleBtn.classList.add('btn-primary');

            editableFields.forEach(field => {
                field.contentEditable = 'true';
                field.style.border = '2px solid #667eea';
                field.style.padding = '8px';
                field.style.borderRadius = '4px';
                field.style.backgroundColor = '#f0f9ff';
            });
        } else {
            // Disable editing and capture values
            editToggleBtn.innerHTML = '<span class="edit-icon">✏️</span> Edit';
            editToggleBtn.classList.remove('btn-primary');
            editToggleBtn.classList.add('btn-secondary');

            editableFields.forEach(field => {
                const fieldName = field.dataset.field;
                const newValue = field.textContent.trim();

                // Parse tags if it's a tags field
                if (fieldName === 'tags') {
                    this.editedValues[fieldName] = newValue.split(',').map(t => t.trim()).filter(t => t);
                } else {
                    this.editedValues[fieldName] = newValue;
                }

                field.contentEditable = 'false';
                field.style.border = '';
                field.style.padding = '';
                field.style.backgroundColor = '';
            });
        }
    }

    async approve() {
        try {
            const approveBtn = this.modal.querySelector('.approve-btn');
            approveBtn.disabled = true;
            approveBtn.textContent = 'Saving...';

            // If there are edited values, use them; otherwise use cached preview
            if (Object.keys(this.editedValues).length > 0) {
                await this.manager.approveEnrichmentWithEdits(this.conversationId, this.editedValues);
            } else {
                await this.manager.approveEnrichment(this.conversationId);
            }

            // Close modal
            this.modal.remove();

            // Update the conversation card in place instead of full reload
            await this.updateConversationCard();

            // Show success message
            if (window.CogRepoUI && window.CogRepoUI.toast) {
                window.CogRepoUI.toast.success('Enrichment saved successfully');
            }

        } catch (error) {
            alert(`Failed to save enrichment: ${error.message}`);
        }
    }

    async updateConversationCard() {
        // Find the conversation card and update it (using correct attribute)
        const card = document.querySelector(`[data-convo-id="${this.conversationId}"]`);
        if (!card) {
            console.warn(`Card not found for conversation ID: ${this.conversationId}`);
            return;
        }

        try {
            // Fetch updated conversation
            const response = await fetch(`/api/conversation/${this.conversationId}`);
            if (!response.ok) throw new Error('Failed to fetch updated conversation');

            const conversation = await response.json();

            // Update title
            const titleEl = card.querySelector('.conversation-title');
            if (titleEl && conversation.generated_title) {
                titleEl.textContent = conversation.generated_title;
            }

            // Update summary if visible
            const summaryEl = card.querySelector('.conversation-summary');
            if (summaryEl && conversation.summary_abstractive) {
                summaryEl.textContent = conversation.summary_abstractive;
            }

            // Update tags
            const tagsContainer = card.querySelector('.conversation-tags');
            if (tagsContainer && conversation.tags) {
                tagsContainer.innerHTML = conversation.tags
                    .map(tag => `<span class="tag">${tag}</span>`)
                    .join('');
            }

            // Add visual feedback
            card.classList.add('just-enriched');
            setTimeout(() => card.classList.remove('just-enriched'), 2000);

        } catch (error) {
            console.error('Failed to update card:', error);
            // Fall back to reload if update fails
            window.location.reload();
        }
    }

    async reject() {
        try {
            await this.manager.rejectEnrichment(this.conversationId);
            this.modal.remove();
        } catch (error) {
            alert(`Failed to discard enrichment: ${error.message}`);
        }
    }
}

// Global instance
export const enrichmentManager = new EnrichmentManager();
