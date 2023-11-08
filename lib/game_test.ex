defmodule GameTest do
  require Nx.Defn

  import Nx.Defn

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
    world[x-1..x+1][y-1..y+1]
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
    _cell = elem(random_cell, 0)
    x = Nx.to_number(elem(random_cell, 1))
    y = Nx.to_number(elem(random_cell, 2))
    world = Nx.indexed_add(world, Nx.tensor([x, y, 1]), 1)

    world
  end

end
