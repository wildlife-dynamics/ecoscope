# Platform SDK Docs

## `vhs` recordings

The gif used in the Getting Started section of these docs was generated 
using the [`vhs` CLI](https://github.com/charmbracelet/vhs). The `.tape`
file used to create the recording is provided here for reference:

<details>

```elixir
Output demo.gif

Require wt-compiler
Require bat

Set Shell "bash"
Set FontSize 28
Set Width 1600
Set Height 600

Type "wt-compiler scaffold init" Sleep 500ms  Enter

Sleep 2s

Down Sleep 1s Enter

Sleep 2s

Type "my_first_workflow" Sleep 1s Enter

Type "My First Workflow" Sleep 1s Enter

Sleep 1s Enter

Type "My Name" Sleep 1s Enter

Down Sleep 500ms Down Sleep 500ms Up Sleep 500ms Up Sleep 500ms Enter

Sleep 1s Enter

Sleep 1s

Type "ls -a1 my_first_workflow" Sleep 500ms Enter

Sleep 3s

Type "bat my_first_workflow/spec.yaml" Sleep 500ms Enter

Sleep 2s

Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms
Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms
Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms Down Sleep 250ms

Sleep 2s

Type "q"

Sleep 5s

```

</details>
