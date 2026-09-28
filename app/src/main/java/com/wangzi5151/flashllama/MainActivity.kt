package com.wangzi5151.flashllama

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.wangzi5151.flashllama.ui.FlashLlamaTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            FlashLlamaTheme {
                AppScreen()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AppScreen(vm: ChatViewModel = viewModel()) {
    val messages by vm.messages.collectAsStateWithLifecycle()
    val params by vm.params.collectAsStateWithLifecycle()
    val info by vm.modelInfo.collectAsStateWithLifecycle()
    val status by vm.status.collectAsStateWithLifecycle()
    val loading by vm.loading.collectAsStateWithLifecycle()
    val generating by vm.generating.collectAsStateWithLifecycle()
    val speed by vm.speed.collectAsStateWithLifecycle()

    var input by remember { mutableStateOf("") }
    var showSettings by remember { mutableStateOf(false) }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> if (uri != null) vm.loadModel(uri) }

    Scaffold(
        topBar = {
            TopAppBar(title = {
                Column {
                    Text("极速GGUF", style = MaterialTheme.typography.titleLarge)
                    if (speed.isNotEmpty()) {
                        Text(speed, style = MaterialTheme.typography.labelSmall)
                    }
                }
            }, actions = {
                IconButton(onClick = { vm.clearChat() }) {
                    Icon(Icons.Default.Delete, contentDescription = "清空对话")
                }
            })
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .imePadding()
        ) {
            ModelCard(
                info = info,
                status = status,
                loading = loading,
                onPick = { picker.launch(arrayOf("*/*")) },
                onUnload = { vm.unload() },
            )

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                TextButton(onClick = { showSettings = !showSettings }) {
                    Icon(Icons.Default.ExpandMore, contentDescription = null)
                    Spacer(Modifier.width(4.dp))
                    Text("参数设置")
                }
                Spacer(Modifier.weight(1f))
                Text(
                    text = "${params.nThreads} 线程 · ctx ${params.nCtx} · t=${params.temperature}",
                    style = MaterialTheme.typography.labelSmall,
                )
            }

            AnimatedVisibility(visible = showSettings) {
                SettingsPanel(params = params, onChange = vm::updateParams)
            }

            Box(modifier = Modifier.weight(1f)) {
                val listState = rememberLazyListState()
                LaunchedEffect(messages.size, messages.lastOrNull()?.content) {
                    if (messages.isNotEmpty()) listState.animateScrollToItem(messages.size - 1)
                }
                if (messages.isEmpty()) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text(
                            "选择模型后即可开始对话",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.outline,
                        )
                    }
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        items(messages) { msg -> MessageBubble(msg) }
                    }
                }
            }

            InputBar(
                value = input,
                onValueChange = { input = it },
                generating = generating,
                enabled = vm.isReady,
                onSend = {
                    val text = input
                    input = ""
                    vm.send(text)
                },
                onStop = { vm.stop() },
            )
        }
    }
}

@Composable
private fun ModelCard(
    info: ModelInfo?,
    status: String,
    loading: Boolean,
    onPick: () -> Unit,
    onUnload: () -> Unit,
) {
    Card(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
        Column(Modifier.padding(12.dp)) {
            if (info == null) {
                Text("未加载模型", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(4.dp))
                Text(status, style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.height(8.dp))
                Button(onClick = onPick, enabled = !loading) {
                    if (loading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary,
                        )
                        Spacer(Modifier.width(8.dp))
                        Text("加载中…")
                    } else {
                        Icon(Icons.Default.FolderOpen, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("选择 GGUF 模型")
                    }
                }
            } else {
                Text(info.name, style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(4.dp))
                Text(info.desc, style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.height(2.dp))
                Text(
                    "参数量 ${formatParams(info.nParams)} · 大小 ${formatBytes(info.sizeBytes)} · " +
                        "上下文 ${info.nCtx}/${info.nCtxTrain} · 词表 ${info.nVocab}",
                    style = MaterialTheme.typography.bodySmall,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    "llama.cpp ${info.llamaVersion} · $status",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline,
                )
                Spacer(Modifier.height(8.dp))
                Row {
                    OutlinedButton(onClick = onPick, enabled = !loading) {
                        Icon(Icons.Default.FolderOpen, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("更换模型")
                    }
                    Spacer(Modifier.width(8.dp))
                    OutlinedButton(onClick = onUnload) { Text("卸载") }
                }
            }
            if (loading) {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun SettingsPanel(
    params: GenParams,
    onChange: ((GenParams) -> GenParams) -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp)
    ) {
        Column(
            Modifier
                .height(300.dp)
                .verticalScroll(rememberScrollState())
                .padding(12.dp)
        ) {
            LabeledSlider("线程数", params.nThreads.toFloat(), 1f, 12f) {
                onChange { p -> p.copy(nThreads = it.toInt()) }
            }
            LabeledSlider("上下文长度", params.nCtx.toFloat(), 512f, 8192f, steps = 14) {
                onChange { p -> p.copy(nCtx = it.toInt()) }
            }
            LabeledSlider("最大生成长度", params.maxTokens.toFloat(), 64f, 2048f, steps = 30) {
                onChange { p -> p.copy(maxTokens = it.toInt()) }
            }
            LabeledSlider("温度", params.temperature, 0f, 2f) {
                onChange { p -> p.copy(temperature = (it * 100).toInt() / 100f) }
            }
            LabeledSlider("Top-K", params.topK.toFloat(), 0f, 100f) {
                onChange { p -> p.copy(topK = it.toInt()) }
            }
            LabeledSlider("Top-P", params.topP, 0f, 1f) {
                onChange { p -> p.copy(topP = (it * 100).toInt() / 100f) }
            }
            LabeledSlider("Min-P", params.minP, 0f, 1f) {
                onChange { p -> p.copy(minP = (it * 100).toInt() / 100f) }
            }
            LabeledSlider("重复惩罚", params.repeatPenalty, 1f, 1.5f) {
                onChange { p -> p.copy(repeatPenalty = (it * 100).toInt() / 100f) }
            }

            Text("KV 缓存精度", style = MaterialTheme.typography.labelLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(0 to "F16", 2 to "Q8_0", 3 to "Q4_0").forEach { (value, label) ->
                    FilterChip(
                        selected = params.kvType == value,
                        onClick = { onChange { p -> p.copy(kvType = value) } },
                        label = { Text(label) },
                    )
                }
            }
            Spacer(Modifier.height(8.dp))

            ToggleRow("锁定内存 (mlock)", params.useMlock) {
                onChange { p -> p.copy(useMlock = it) }
            }
            ToggleRow("套用对话模板", params.useChatTemplate) {
                onChange { p -> p.copy(useChatTemplate = it) }
            }
            Text(
                "修改上下文/KV/线程数后需重新加载模型生效",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline,
            )
        }
    }
}

@Composable
private fun LabeledSlider(
    label: String,
    value: Float,
    min: Float,
    max: Float,
    steps: Int = 0,
    onChange: (Float) -> Unit,
) {
    Column(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth()) {
            Text(label, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.weight(1f))
            Text(
                if (value >= 100) value.toInt().toString() else value.toString(),
                style = MaterialTheme.typography.labelSmall,
            )
        }
        Slider(
            value = value.coerceIn(min, max),
            onValueChange = onChange,
            valueRange = min..max,
            steps = steps,
        )
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.weight(1f))
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun MessageBubble(msg: ChatMessage) {
    val isUser = msg.role == "user"
    val bubbleColor = if (isUser) MaterialTheme.colorScheme.primary
    else MaterialTheme.colorScheme.surfaceVariant
    val textColor = if (isUser) MaterialTheme.colorScheme.onPrimary
    else MaterialTheme.colorScheme.onSurfaceVariant

    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Box(
            Modifier
                .fillMaxWidth(0.9f)
                .background(bubbleColor, RoundedCornerShape(12.dp))
                .padding(10.dp)
        ) {
            Column {
                Text(
                    text = if (msg.content.isEmpty() && msg.streaming) "…" else msg.content,
                    color = textColor,
                    fontFamily = if (isUser) FontFamily.Default else FontFamily.Monospace,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}

@Composable
private fun InputBar(
    value: String,
    onValueChange: (String) -> Unit,
    generating: Boolean,
    enabled: Boolean,
    onSend: () -> Unit,
    onStop: () -> Unit,
) {
    Row(
        Modifier
            .fillMaxWidth()
            .padding(8.dp),
        verticalAlignment = Alignment.Bottom,
    ) {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.weight(1f),
            placeholder = { Text(if (enabled) "输入消息…" else "请先加载模型") },
            maxLines = 4,
            enabled = enabled && !generating,
        )
        Spacer(Modifier.width(8.dp))
        if (generating) {
            Button(onClick = onStop) {
                Icon(Icons.Default.Stop, contentDescription = "停止")
                Spacer(Modifier.width(4.dp))
                Text("停止")
            }
        } else {
            Button(onClick = onSend, enabled = enabled && value.isNotBlank()) {
                Icon(Icons.Default.Send, contentDescription = "发送")
            }
        }
    }
}
