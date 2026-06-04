<template>
  <CompareModePlaceholder v-if="ui.viewMode === 'compare'" />
  <DetailPanel v-else />
</template>

<script setup lang="ts">
import { useUIStore } from '@/stores/ui';

import CompareModePlaceholder from './CompareModePlaceholder.vue';
import DetailPanel from './DetailPanel.vue';

const ui = useUIStore();
</script>
