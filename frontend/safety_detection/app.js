const { createApp } = Vue;

createApp({
    data() {
        return {
            detecting: false,
            personDetected: false,
            analyzing: false,
            result: null,
            logs: [],
            records: []
        };
    },
    computed: {
        statusText() {
            if (this.analyzing) return '分析中';
            if (this.personDetected) return '已检测到';
            return '正常';
        },
        statusClass() {
            if (this.analyzing) return 'active';
            if (this.personDetected) return 'warning';
            return 'normal';
        }
    },
    methods: {
        async fetchStatus() {
            try {
                const res = await fetch('/status');
                const data = await res.json();

                this.detecting = data.detecting || false;
                this.personDetected = data.person_detected || false;
                this.analyzing = data.analyzing || false;
                this.result = data.result || null;
                this.logs = data.logs || [];
                this.records = data.records || [];
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        },
        getActionClass(action) {
            if (action === 'enter') return 'alert';
            if (action === 'leave') return 'leave';
            return 'safe';
        },
        getActionIcon(action) {
            if (action === 'enter') return '🚨';
            if (action === 'leave') return '🚪';
            return '✅';
        },
        getActionText(action) {
            if (action === 'enter') return '进入电梯';
            if (action === 'leave') return '离开电梯';
            return '无动作';
        },
        getRecordActionClass(record) {
            if (!record) return 'pending';
            if (record.action === null || record.action === undefined) return 'pending';
            if (record.action === 'enter') return 'alert';
            if (record.action === 'leave') return 'leave';
            return 'safe';
        },
        getRecordStatusClass(record) {
            if (!record) return 'pending';
            if (record.entry === null || record.entry === undefined) return 'pending';
            return record.entry ? 'alert' : 'safe';
        }
    },
    mounted() {
        this.fetchStatus();
        setInterval(() => this.fetchStatus(), 3000);
    }
}).mount('#app');