package dao;

public class MySqlDaoFactory implements DaoFactory{
    public ComputerDao createComputerDao(){
        return new MySqlComputerDao();
    }
    public StaffDao createStaffDao(){
        return new SqlServerStaffDao();
    }
}
