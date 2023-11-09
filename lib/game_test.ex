defmodule GameTest do
  require Nx.Defn

  import Nx.Defn

  Nx.Defn.default_options(compiler: EXLA, client: :cuda)
  Nx.global_default_backend({EXLA.Backend, client: :cuda})


  @moduledoc """
  Documentation for `LearningElixir`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> LearningElixir.hello()
      :world

  """

  def create_world(num) do
    cell_info = Nx.tensor([0, 0, 50], names: [:cell_info], type: {:u, 32})
    world = Nx.broadcast(cell_info, {num, num, 3})
    world = Nx.tensor(world, names: [:x, :y, :cell_info], type: {:u, 32})
    indices = Nx.iota({num, num}, names: [:x, :y], type: {:u, 32})
    indices = Nx.new_axis(indices, 2, :cell_info)
    world = Nx.put_slice(world, [0, 0, 0], indices)
    # world = Nx.concatenate([world, indices])
    world
  end

  defn get_cell(world, x, y) do
    world[x][y]
  end

  def get_neighbours(world, x, y) do
    shape = Nx.shape(world)
    rows = elem(shape, 0)
    IO.puts("rows: #{rows}")
    columns = elem(shape, 1)
    IO.puts("columns: #{columns}")
    cond do
      # top left corner
      x == 0 && y == 0 ->
        IO.puts("top left corner")
        neighbours = Nx.slice(world, [x, y, 0], [2, 2, 3])
        {neighbours, 0, 0}

        # top edge
      x == 0 && y > 0 && y < columns-1 ->
        IO.puts("top edge")
        neighbours = Nx.slice(world, [x, y-1, 0], [2, 3, 3])
        {neighbours, 0, 1}

        # top right corner
      x == 0 && y == columns-1 ->
        IO.puts("top right corner")
        neighbours = Nx.slice(world, [x, y-1, 0], [2, 2, 3])
        {neighbours, 1, 1}

        # right edge
      x > 0 && x < rows-1 && y == columns-1 ->
        IO.puts("right edge")
        neighbours = Nx.slice(world, [x-1, y-1, 0], [3, 2, 3])
        {neighbours, 1, 1}

        # bottom right corner
      x == rows-1 && y == columns-1 ->
        IO.puts("bottom right corner")
        neighbours = Nx.slice(world, [x-1, y-1, 0], [2, 2, 3])
        {neighbours, 1, 1}

        # bottom edge
      x == rows-1 && y > 0 && y < columns-1 ->
        IO.puts("bottom edge")
        neighbours = Nx.slice(world, [x-1, y-1, 0], [2, 3, 3])
        {neighbours, 1, 1}

        # bottom left corner
      x == rows-1 && y == 0 ->
        IO.puts("bottom left corner")
        neighbours = Nx.slice(world, [x-1, y, 0], [2, 2, 3])
        {neighbours, 1, 0}

        # left edge
      x > 0 && x < rows-1 && y == 0 ->
        IO.puts("left edge")
        neighbours = Nx.slice(world, [x-1, y, 0], [3, 2, 3])
        {neighbours, 1, 0}

        # middle (all other cases)
      x > 0 && x < rows-1 && y > 0 && y < columns-1 ->
        IO.puts("middle")
        neighbours = Nx.slice(world, [x-1, y-1, 0], [3, 3, 3])
        {neighbours, 1, 1}
    end
  end

  defn choose_random_cell(world) do
    shape = Nx.shape(world)
    rows = elem(shape, 0)
    columns = elem(shape, 1)
    key = Nx.Random.key(0)
    {row, _new_key} = Nx.Random.randint(key, 1, rows)
    {column, _new_key} = Nx.Random.randint(key, 1, columns)

    {world[row][column], row, column}
  end

  def pollinate_random(world) do
    random_cell = choose_random_cell(world)
    cell = elem(random_cell, 0)

    x = Nx.to_number(elem(random_cell, 1))
    y = Nx.to_number(elem(random_cell, 2))

    IO.puts(Nx.to_number(cell[1]))
    cond do
      Nx.to_number(cell[1]) == 0 ->
        world = Nx.indexed_add(world, Nx.tensor([x, y, 1]), 1)
        world
      Nx.to_number(cell[1]) == 1 ->
        world
    end
    #     world = Nx.indexed_add(world, Nx.tensor([x, y, 1]), 1)
    #     world
    # world
  end

  def update_cell(world, x, y) do
    # get cell
    cell = get_cell(world, x, y)
    cell_num = Nx.to_number(cell[0])
    is_allive = Nx.to_number(cell[1])
    energy_amt = Nx.to_number(cell[2])
    {neighbours, _x_offset, _y_offset} = get_neighbours(world, x, y)
    energy_per_round = 10
    # check status
    cond do
      # 0 means cell is dead
      is_allive == 0 ->
        # check status of neighbours
        # we can use the mode function to get the most common value
        # (if there are multiple organism types)
        sum = Nx.sum(neighbours[cell_info: 1])
        cond do
          # if there is at least one neighbour, cell is born
          sum > 0 ->
            cell = Nx.tensor([cell_num, 1, energy_amt], names: [:cell_info], type: {:u, 32})
            cell

          # if there are no neighbours, cell stays dead
          sum == 0 ->
            cell
        end

      # 0 < means cell is alive / has organism in it
      is_allive == 1 ->
        # eat x energy_amt
        energy_amt = Nx.to_number(cell[2])
        cond do
          # check if cell has enough energy
          energy_amt >= energy_per_round ->
            # eat x energy_amt
            cell = Nx.tensor([cell_num, 1, energy_amt - energy_per_round], names: [:cell_info], type: {:u, 32})
            cell

            # if not, cell dies
          energy_amt < energy_per_round ->
            cell = Nx.tensor([cell_num, 0, 0], names: [:cell_info], type: {:u, 32})
            cell
        end
    end
  end

  def update_world(world, x, y, cell) do
    IO.inspect(cell)
    # index does not have to be updated
    world = Nx.indexed_put(world, Nx.tensor([x, y, 1]), cell[1])
    world = Nx.indexed_put(world, Nx.tensor([x, y, 2]), cell[2])
    world
  end
end
