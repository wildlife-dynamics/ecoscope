# Getting Started

The following walkthrough will help you get up and running with 

---

## Prerequisites

- First, [install `pixi`](https://pixi.prefix.dev/latest/installation/)
and [`uv`](https://docs.astral.sh/uv/getting-started/installation/#installing-uv)
if you do not have both already.

- Then, install `wt-compiler`:

    ??? note "Note on `--run-post-link-scripts`"
        The `--run-post-link-scripts` flag is necessary because, in order to generate a visual
        representation of the workflow DAG, the `wt-compiler compile` command depends on the `dot` executable
        having been initialized post-install via the `dot -c` command. Setting the `--run-post-link-scripts`
        flag triggers this initialization automatically. Setting this flag does imply allowing the package
        manager to [run (potentially insecure) arbitrary scripts](https://pixi.prefix.dev/v0.62.2/reference/pixi_configuration/#run-post-link-scripts).
        If you prefer to omit this flag, then after you have installed `wt-compiler`, you may
        separately run `$HOME/.pixi/envs/wt-compiler/bin/dot -c` to initialize `dot`.

    ```console
    $ pixi global install \
    -c https://prefix.dev/ecoscope-workflows \
    -c conda-forge \
    wt-compiler \
    --run-post-link-scripts
    ```

- Third, install the the Ecoscope Wizard Providers plugin for `wt-compiler`:

    ```console
    $ pixi global add --environment wt-compiler ecoscope-wizard-providers
    ```

- Finally, [download and install the Ecoscope Desktop App](https://app.ecoscope.io/download) if you do not have it already. 

---

## Step 1 - Scaffold a new workflow

In a clean directory on your machine, run: 

```console
$ wt-compiler scaffold init
```

From the selection of Ecoscope wizard providers presented to you interactively, choose `events-map-toy`.
Then, follow the remainder of the prompts until the wizard exits, and you will now see that the scaffold of
your workflow has been created:

```
$ ls -la 
...
```

## Step 2 - Compile the scaffold into a workflow template

From the new directory, run:

```console
$ wt-compiler compile --spec=spec.yaml --install
```

You will now see:

```console
$ ls -la
...
```

## Step 3 - Load the template into Ecoscope Desktop

...

## Step 4 - Create and run a workflow using the template

### Setup a Data Source

...

### Configure the workflow

...

### Run the workflow

...

## Step 4 - Change a parameter in `spec.yaml`, recompile, and re-run

Here you will get your first taste of the iterative workflow development process.

Let's change a simple parameter in the `spec.yaml` for this toy example, then we
will recompile the workflow, and see the changed result by re-running the updated
workflow in the Ecoscope Desktop app.

## Next step - Adding your own tasks

This intro guide has walked you through the basic iterative flow of workflow development
for Ecoscope Desktop. The next step, will be adding your own custom logic (or, "tasks")
to customize the behavior of the workflow.
